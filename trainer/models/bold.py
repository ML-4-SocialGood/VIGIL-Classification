import copy
import os

import timm
import torch
import torch.nn as nn
from clip import clip
from tabulate import tabulate
from torch.nn import functional as F

from metrics import compute_accuracy
from optim import build_lr_scheduler, build_optimizer
from trainer import MODEL_REGISTRY, Trainer
from utils import PROMPT_TEMPLATES
from utils.tools import count_num_parameters
from ops import GradNorm, UncertaintyWeighting, ParetoMTL


class Adapter(nn.Module):
    def __init__(self, channel_in, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_in, channel_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in // reduction, channel_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        if self.cfg.MODEL.BOLD.BACKBONE == "RN50":
            adapter_dim = 1024
        elif self.cfg.MODEL.BOLD.BACKBONE == "ViT-B/32":
            adapter_dim = 512
        self.domain_aware_adapters = nn.ModuleList(
            [
                Adapter(adapter_dim, 4).to(clip_model.dtype)
                for i in range(len(cfg.DATASET.SOURCE_DOMAINS))
            ]
        )
        self.dtype = clip_model.dtype

        prompt_template = PROMPT_TEMPLATES[cfg.DATASET.NAME]
        prompts = {}
        # prompts_classic = [
        #     prompt_template.format(class_name.replace("_", " "))
        #     for class_name in class_names
        # ]
        # prompts["classic"] = prompts_classic
        for domain in cfg.DATASET.SOURCE_DOMAINS:
            prompts[domain] = [
                prompt_template.format(
                    domain.replace("_", " ") + " " + class_name.replace("_", " ")
                )
                for class_name in class_names
            ]
        self.text_features = {}
        for domain, prompts_domain in prompts.items():
            prompts_domain_tokenized = torch.cat(
                [clip.tokenize(prompt) for prompt in prompts_domain]
            )
            prompts_domain_tokenized = prompts_domain_tokenized.to(
                torch.cuda.current_device()
            )
            with torch.no_grad():
                self.text_features[domain] = clip_model.encode_text(
                    prompts_domain_tokenized
                )
                self.text_features[domain] = self.text_features[
                    domain
                ] / self.text_features[domain].norm(dim=-1, keepdim=True)

    def forward(self, image, domain_labels=None):
        adapter_ratio = 0.2
        invariant_image_features = self.image_encoder(image.type(self.dtype))

        adapter_features = []
        for image_feature, domain_label in zip(invariant_image_features, domain_labels):
            adapter_features.append(
                self.domain_aware_adapters[domain_label](image_feature)
            )
        adapter_features = torch.vstack(adapter_features)
        specific_image_features = (
            adapter_ratio * adapter_features
            + (1 - adapter_ratio) * invariant_image_features
        )
        specific_image_features = (
            specific_image_features / specific_image_features.norm(dim=-1, keepdim=True)
        )

        logit_scale = self.logit_scale.exp()
        logits_domain = {}
        for domain, text_feature in self.text_features.items():
            logits_domain[domain] = (
                logit_scale * specific_image_features @ text_feature.t()
            )
        logits = torch.cat(list(logits_domain.values()), dim=1)

        return invariant_image_features, specific_image_features, logits


@MODEL_REGISTRY.register()
class BOLD(Trainer):
    def build_model(self):
        print("Loading CLIP Backbone: {}".format(self.cfg.MODEL.BOLD.BACKBONE))
        clip_model, _ = clip.load(
            self.cfg.MODEL.BOLD.BACKBONE,
            device=self.device,
            download_root=os.path.abspath(os.path.expanduser("data")),
        )

        print("Building Teacher Model")
        self.teacher = CustomCLIP(
            self.cfg, self.data_manager.dataset.class_names, clip_model
        )

        print("Turning Off Gradients in Image and Text Encoder")
        for name, param in self.teacher.named_parameters():
            if "adapter" not in name:
                param.requires_grad_(False)

        # Double check
        enabled_params = set()
        for name, param in self.teacher.named_parameters():
            if param.requires_grad:
                enabled_params.add(name)
        print("Parameters to be updated: {}".format(enabled_params))

        self.teacher.to(self.device)

        # NOTE: Only Give domain_aware_adapters to the Optimizer
        self.optimizer_teacher = build_optimizer(
            self.teacher.domain_aware_adapters, self.cfg.OPTIM
        )
        self.lr_scheduler_teacher = build_lr_scheduler(
            self.optimizer_teacher, self.cfg.OPTIM
        )

        self.model_registeration(
            "BOLD_Teacher",
            self.teacher.domain_aware_adapters,
            self.optimizer_teacher,
            self.lr_scheduler_teacher,
        )

        print("Building Student Model")
        teacher_network = self.cfg.MODEL.BOLD.BACKBONE
        student_network = self.cfg.MODEL.BOLD.STUDENT_NETWORK

        self.student_model = timm.create_model(
            self.cfg.MODEL.BOLD.STUDENT_NETWORK,
            pretrained=True,
            num_classes=self.num_classes,
        )

        if not (teacher_network == "ViT-B/32" and student_network == "resnet18"):
            if teacher_network == "ViT-B/32":
                self.student_model.projection_layer = nn.Linear(
                    self.student_model.fc.in_features, 512, bias=True
                )
            elif teacher_network == "RN50":
                self.student_model.projection_layer = nn.Linear(
                    self.student_model.fc.in_features, 1024, bias=True
                )
            else:
                raise NotImplementedError

            del self.student_model.fc
            self.student_model.fc = nn.Linear(
                self.student_model.projection_layer.out_features,
                self.num_classes,
                bias=True,
            )
        self.student_model.to(self.device)

        self.gradNorm = GradNorm(loss_weights=[0.5, 0.5], num_tasks=2)
        self.uncertainty_weighting = UncertaintyWeighting()
        self.uncertainty_weighting.to(self.device)
        self.pareto_mtl = ParetoMTL(num_tasks=2)

        # self.optimizer_student = build_optimizer(self.student_model, self.cfg.OPTIM)
        self.optimizer_student = build_optimizer(
            list(self.student_model.parameters())
            + list(self.uncertainty_weighting.parameters()),
            self.cfg.OPTIM,
        )
        self.lr_scheduler_student = build_lr_scheduler(
            self.optimizer_student, self.cfg.OPTIM
        )
        self.model_registeration(
            "BOLD_Student",
            self.student_model,
            self.optimizer_student,
            self.lr_scheduler_student,
        )

        self.distillation_loss_weight = self.cfg.MODEL.BOLD.LOSS_WEIGHT.DISTILLATION
        self.classification_loss_weight = self.cfg.MODEL.BOLD.LOSS_WEIGHT.CLASSIFICATION
        self.temperature = self.cfg.MODEL.BOLD.TEMPERATURE

        print("Distillation_Loss_Weight: {}".format(self.distillation_loss_weight))
        print("Classification_Loss_Weight: {}".format(self.classification_loss_weight))
        print("Temperature: {}".format(self.temperature))

        model_parameters_table = [
            ["Model", "# Parameters"],
            [
                "Teacher Adapter",
                f"{count_num_parameters(self.teacher.domain_aware_adapters[0]) * len(self.cfg.DATASET.SOURCE_DOMAINS):,}",
            ],
            ["Student", f"{count_num_parameters(self.student_model):,}"],
        ]
        print(tabulate(model_parameters_table))

    def forward_backward(self, batch_data):
        image, class_label, domain_label = self.parse_batch_train(batch_data)
        copy_student = copy.deepcopy(self.student_model)
        copy_teacher = copy.deepcopy(self.teacher)

        invariant_image_features, specific_image_features, logits_teacher = (
            self.teacher(image, domain_label)
        )
        student_image_features = copy_student.forward_features(image)
        student_image_features = copy_student.global_pool(student_image_features)
        if not (
            self.cfg.MODEL.BOLD.BACKBONE == "ViT-B/32"
            and self.cfg.MODEL.BOLD.STUDENT_NETWORK == "resnet18"
        ):
            student_image_features = copy_student.projection_layer(
                student_image_features
            )
        logits_teacher = torch.split(logits_teacher, self.num_classes, dim=1)
        loss_teacher_ce = F.cross_entropy(
            logits_teacher[0], class_label, reduction="none"
        )
        for i in range(len(domain_label)):
            for idx, domain_output in enumerate(logits_teacher[1:]):
                if domain_label[i] == idx:
                    domain_loss = F.cross_entropy(domain_output, class_label)
                else:
                    domain_loss = -0.1 * F.cross_entropy(domain_output, class_label)
                loss_teacher_ce[i] = loss_teacher_ce[i] + domain_loss
        loss_teacher_ce = loss_teacher_ce.mean()
        loss_specific_distillation = (
            F.kl_div(
                F.log_softmax(student_image_features, dim=1),
                F.softmax(specific_image_features, dim=1),
                reduction="batchmean",
            )
            * self.temperature
            * self.temperature
        )
        loss_teacher = loss_teacher_ce + loss_specific_distillation
        self.model_backward_and_update(loss_teacher, model_names="BOLD_Teacher")
        invariant_image_features, specific_image_features, logits_teacher = (
            copy_teacher(image, domain_label)
        )
        student_image_features = self.student_model.forward_features(image)
        student_image_features = self.student_model.global_pool(student_image_features)
        if not (
            self.cfg.MODEL.BOLD.BACKBONE == "ViT-B/32"
            and self.cfg.MODEL.BOLD.STUDENT_NETWORK == "resnet18"
        ):
            student_image_features = self.student_model.projection_layer(
                student_image_features
            )
        logits_student = self.student_model.fc(student_image_features)
        loss_student_ce = F.cross_entropy(logits_student, class_label)
        loss_invariant_distillation = (
            F.kl_div(
                F.log_softmax(student_image_features, dim=1),
                F.softmax(invariant_image_features, dim=1),
                reduction="batchmean",
            )
            * self.temperature**2
        )
        loss_specific_distillation = (
            F.kl_div(
                F.log_softmax(student_image_features, dim=1),
                F.softmax(specific_image_features, dim=1),
                reduction="batchmean",
            )
            * self.temperature**2
        )

        # grads_invariant = torch.autograd.grad(
        #     loss_invariant_distillation,
        #     self.student_model.parameters(),
        #     retain_graph=True,
        #     allow_unused=True,
        # )
        # grads_specific = torch.autograd.grad(
        #     loss_specific_distillation,
        #     self.student_model.parameters(),
        #     retain_graph=True,
        #     allow_unused=True,
        # )

        # grads_invariant_flat = torch.cat(
        #     [g.view(-1) for g in grads_invariant if g is not None]
        # )
        # grads_specific_flat = torch.cat(
        #     [g.view(-1) for g in grads_specific if g is not None]
        # )

        # dot_product = torch.dot(grads_invariant_flat, grads_specific_flat)
        # norm_invariant = torch.norm(grads_invariant_flat)
        # norm_specific = torch.norm(grads_specific_flat)
        # cosine_similarity = dot_product / (norm_invariant * norm_specific)

        # print(f"Cosine Similarity between gradients: {cosine_similarity.item()}")

        # GradNorm
        # grads = [
        #     torch.autograd.grad(
        #         loss_invariant_distillation,
        #         self.student_model.parameters(),
        #         retain_graph=True,
        #         allow_unused=True,
        #     ),
        #     torch.autograd.grad(
        #         loss_specific_distillation,
        #         self.student_model.parameters(),
        #         retain_graph=True,
        #         allow_unused=True,
        #     ),
        # ]

        # self.gradNorm.update_weights(grads=grads)
        # dynamic_weights = self.gradNorm.get_weights()
        # print("Dynamic Weights: {}".format(dynamic_weights))

        # Uncertainty Weighting
        inv_spc_loss = self.uncertainty_weighting(
            loss_invariant_distillation, loss_specific_distillation
        )
        loss_student = loss_student_ce + inv_spc_loss
        # print("sigma1: {}".format(torch.exp(self.uncertainty_weighting.log_sigma1)))
        # print("sigma2: {}".format(torch.exp(self.uncertainty_weighting.log_sigma2)))

        # ParetoMTL
        # losses = [loss_invariant_distillation, loss_specific_distillation]
        # params = list(self.student_model.parameters())
        # pareto_grad = self.pareto_mtl.calculate_pareto_grad(losses, params)

        # with torch.no_grad():
        #     idx = 0
        #     for param in params:
        #         if param.grad is not None:
        #             param.grad.copy_(
        #                 pareto_grad[idx: idx + param.numel()].view_as(param)
        #             )
        #             idx += param.numel()

        # self.optimizer_student.step()
        # self.optimizer_student.zero_grad()

        # loss_student = (
        #     loss_student_ce + loss_invariant_distillation + loss_specific_distillation
        # )
        self.model_backward_and_update(loss_student, model_names="BOLD_Student")

        loss_summary = {
            "loss_teacher_ce": loss_teacher_ce.item(),
            "loss_student_ce": loss_student_ce.item(),
            "loss_invariant": loss_invariant_distillation.item(),
            "loss_specific": loss_specific_distillation.item(),
            "loss_teacher": loss_teacher.item(),
            "loss_student": loss_student.item(),
            "acc_student": compute_accuracy(logits_student, class_label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary, [
            torch.exp(self.uncertainty_weighting.log_sigma1).cpu().detach().numpy(),
            torch.exp(self.uncertainty_weighting.log_sigma2).cpu().detach().numpy(),
        ]

    def parse_batch_train(self, batch_data):
        input_data = batch_data["img"].to(self.device)
        class_label = batch_data["class_label"].to(self.device)
        domain_label = batch_data["domain_label"].to(self.device)
        return input_data, class_label, domain_label

    def model_inference(self, input_data):
        if (
            self.cfg.MODEL.BOLD.BACKBONE == "ViT-B/32"
            and self.cfg.MODEL.BOLD.STUDENT_NETWORK == "resnet18"
        ):
            return self.student_model(input_data)
        else:
            image_features = self.student_model.forward_features(input_data)
            image_features = self.student_model.global_pool(image_features)
            image_features = self.student_model.projection_layer(image_features)
            return self.student_model.fc(image_features)
