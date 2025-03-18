from train_scripts.modeling.sana import SanaTransformer2DModelWithContext
from diffusers import SanaPipeline
from train_scripts.modeling.pipeline import sana_pipeline_inference
import torch
from transformers import CLIPImageProcessor,SiglipVisionModel,SiglipImageProcessor

pretrained_model_name_or_path = '/data0/jacklishufan/Sana_1600M_1024px_MultiLing_diffusers'
image_feature_dim = 1024

extra_model_kwargs = dict(
                reflection=True,
                reflection_mode='decoupled',
                vision_feature_dim=image_feature_dim,
                num_reflection_transformer_layers=2,
                low_cpu_mem_usage=False
)
device = 'cuda'
model_path = pretrained_model_name_or_path
model_path = '/data0/jacklishufan/trained-sana-lora/checkpoint-5000'
model_path = '/data0/jacklishufan/trained-sana-reflection-resume/checkpoint-5000'
model_path = '/data0/jacklishufan/trained-sana-reflection-decoupled/checkpoint-5000'

transformer = SanaTransformer2DModelWithContext.from_pretrained(
            model_path, subfolder="transformer", #revision=args.revision, variant=args.variant,
            **extra_model_kwargs,
).to(device).to(torch.bfloat16)

clip  = 'google/siglip-large-patch16-384'
image_encoder = SiglipVisionModel.from_pretrained(clip)
encoder_image_processor = SiglipImageProcessor.from_pretrained(clip)
image_encoder.to(device).to(torch.bfloat16)

pipeline = SanaPipeline.from_pretrained(
    pretrained_model_name_or_path,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to('cuda')
prompt = 'a japanese girl with sailor uniform and a sword in her hand, wearing black tights"'
prompt = 'a photo of a knife above a toaster'
feedback_image = 'examples/case1/feedback.png'
feedack_text = 'There is no toaster in image'
# imgs = pipeline(
#         prompt=prompt,
#         height=1024,
#         width=1024,
#         guidance_scale=4.5,
#         num_inference_steps=20,
#         #generator=torch.Generator(device="cuda").manual_seed(42),
#     )[0]
# imgs[0].save('sana-2.jpg')

imgs = sana_pipeline_inference(
        pipeline,
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.5,
        num_inference_steps=20,
        do_reflection=True,
        image_encoder=image_encoder,
        encoder_image_processor=encoder_image_processor,
        n_reflection_tokens_per_img=64,
        feedback_imgs = [feedback_image,feedback_image],
        feedback_texts= [feedack_text,feedack_text]
    )
imgs[0][0].save('sana-ref-out.jpg')
breakpoint()