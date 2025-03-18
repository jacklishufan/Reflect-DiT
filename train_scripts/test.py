from train_scripts.modeling.sana import SanaTransformer2DModelWithContext
from diffusers import SanaPipeline
import torch

pretrained_model_name_or_path = '/data0/jacklishufan/Sana_1600M_1024px_MultiLing_diffusers'
image_feature_dim = 1024

extra_model_kwargs = dict(
                reflection=True,
                vision_feature_dim=image_feature_dim,
                num_reflection_transformer_layers=2,
                low_cpu_mem_usage=False
    )

model_path = pretrained_model_name_or_path
model_path = '/data0/jacklishufan/trained-sana-lora/checkpoint-5000'
transformer = SanaTransformer2DModelWithContext.from_pretrained(
            model_path, subfolder="transformer", #revision=args.revision, variant=args.variant,
            **extra_model_kwargs,
).cuda()

pipeline = SanaPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to('cuda')
prompt = 'a japanese girl with sailor uniform and a sword in her hand, wearing black tights"'

imgs = pipeline(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=20,
        #generator=torch.Generator(device="cuda").manual_seed(42),
    )[0]
imgs[0].save('sana-2.jpg')
breakpoint()