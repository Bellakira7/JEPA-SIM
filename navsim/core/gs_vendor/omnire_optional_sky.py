"""Adapt the pinned native RGB/depth loss to genuinely absent sky labels."""
import inspect
import textwrap


def install():
    from models.trainers import base
    trainer=base.BasicTrainer
    original=trainer.compute_losses
    if getattr(original,'_openscene',False):
        return
    source=textwrap.dedent(inspect.getsource(original))
    changes={
        'torch.ones_like(image_infos["sky_masks"])':
            'torch.ones_like(image_infos["pixels"][..., 0])',
        'gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask':
            'gt_occupied_mask = None if image_infos.get("sky_masks") is None else (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask',
        'if self.sky_opacity_loss_fn is not None:':
            'if self.sky_opacity_loss_fn is not None and self.losses_dict.mask.w > 0:\n        if gt_occupied_mask is None:\n            raise ValueError("Sky supervision requested without annotations")',
    }
    for old,new in changes.items():
        if source.count(old)!=1:
            raise RuntimeError('Pinned native OmniRe loss changed: '+old)
        source=source.replace(old,new)
    namespace={}
    exec(compile(source,__file__,'exec'),vars(base),namespace)
    adapted=namespace['compute_losses']
    adapted._openscene=True
    trainer.compute_losses=adapted
