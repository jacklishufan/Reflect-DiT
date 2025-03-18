import sys
import os
# set import dir to be the parent folder
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(parent)
# exit()
sys.path.append(parent)
import torch
from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3Pipeline,
)


import os
from diffusers import SanaPipeline,SanaTransformer2DModel
from train_scripts.modeling.sana import SanaTransformer2DModelWithContext
from transformers import CLIPTextModelWithProjection,CLIPVisionModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast,CLIPImageProcessor
import argparse
import pandas as pd
from argparse import ArgumentParser
import accelerate
from PIL import Image, ImageOps
import numpy as np
from train_scripts.geneval_utils import SegmentationFeedback,QwenFeedback
from train_scripts.modeling.pipeline import sana_pipeline_inference
from train_scripts.dpg_util import DPGFeedback,prepare_dpg_data

class DummyFeedback:
    
    def __init__(self,device):
        pass
    
    def evaluate_image(self,image_path,metadata):
       return dict(
            correct=1,
            text_feedback='',
            prompt=metadata['prompt'],
        )
       

def read_json(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


import glob

parser = argparse.ArgumentParser()
ckpts =''
base_model_path='legacy_not_used'
text_vae_path='legacy_not_used'

parser.add_argument('--ckpt',type=str,default=ckpts)
parser.add_argument('--output_path',type=str,default='outputs_geneval')
parser.add_argument('--cfg',type=str,default='7')
parser.add_argument('--name',type=str,default='none')
parser.add_argument('--base_model_path',type=str,default=base_model_path)
parser.add_argument('--text_vae_path',type=str,default=text_vae_path)
parser.add_argument('--shift',type=float,default=3.0)
parser.add_argument('--ema',action='store_true')
parser.add_argument('-n','--num_samples',type=int,default=20)
parser.add_argument('-k','--n_reflection_tokens_per_img',type=int,default=64)
parser.add_argument('--cont',action='store_true')
parser.add_argument('--drop_feedback',action='store_true')
parser.add_argument('--no_reflection',action='store_true')
parser.add_argument('--cleanup',action='store_true')
parser.add_argument('--n_feedback',type=int,default=3)
parser.add_argument('--dataset',type=str,choices=['geneval','dpg','custom'])
parser.add_argument('--add_prefix_feedback',action='store_true')
parser.add_argument('--fmt',type=str,default='png')

# vlm_path
parser.add_argument('--vlm_path',type=str,default='')


parser.add_argument('-i','--annotation',default='geneval/prompts/evaluation_metadata.jsonl',type=str)

args = parser.parse_args()

def build_pipeline(accelerator,args):
    base_model_path = args.base_model_path
    transformer = SanaTransformer2DModelWithContext.from_pretrained(args.ckpt,subfolder="transformer",device='cpu')
    transformer.to(torch.bfloat16)
    extra_kwargs = dict(transformer=transformer)
    pipeline = SanaPipeline.from_pretrained(
        base_model_path,
        device='cpu',
        torch_dtype=torch.bfloat16,
        **extra_kwargs,
    )
    pipeline = pipeline.to(accelerator.device)
    def _generate_one_image(x,**kwargs):
        return pipeline(
            prompt=x,
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=20,
            **kwargs
        )
    return pipeline,_generate_one_image


from matplotlib import pyplot as plt
from PIL import Image

def center_crop_and_resize(image_path, desired_size):
    # Open the image
    image = Image.open(image_path)
    
    # Get dimensions
    width, height = image.size
    
    # Calculate the size of the largest square
    new_side = min(width, height)
    
    # Calculate the cropping box
    left = (width - new_side) / 2
    top = (height - new_side) / 2
    right = (width + new_side) / 2
    bottom = (height + new_side) / 2
    
    # Crop the image to the largest square
    image = image.crop((left, top, right, bottom))
    
    # Resize the image to the desired size
    image = image.resize((desired_size, desired_size))
    
    return image

from transformers import CLIPImageProcessor,SiglipVisionModel,SiglipImageProcessor
def subsample_by_unique_reason(data_list, sample_size=3):
    """
    Selects up to `sample_size` samples with unique reasons from the given list.
    
    :param data_list: List of dictionaries with 'path' and 'reason' keys
    :param sample_size: Number of unique samples to select (default is 3)
    :return: List of selected dictionaries
    """
    reason_map = {}
    for item in data_list:
        reason = item['reason']
        if reason not in reason_map:
            reason_map[reason] = item
    
    unique_samples = np.array(list(reason_map.values()))
    return list(np.random.choice(unique_samples, min(sample_size, len(unique_samples)), replace=False))

accelerator = accelerate.Accelerator()
pipeline,generate_one_image = build_pipeline(accelerator,args)
local_rank = accelerator.process_index
world_size = accelerator.num_processes
ROOT=os.path.join(args.output_path,f'gen_eval_{args.name}/')
if accelerator.is_main_process:
    if os.path.exists(ROOT) and args.cleanup:
        import shutil
        shutil.rmtree(ROOT)
accelerator.wait_for_everyone()

os.makedirs(ROOT,exist_ok=True)
torch.cuda.set_device(accelerator.device)

if args.dataset == 'geneval':
    verifier = SegmentationFeedback(accelerator.device)
elif args.dataset == 'dpg':
    verifier = DPGFeedback(device=f'cuda:{local_rank}')
else: # custom
    verifier = DummyFeedback(device=f'cuda:{local_rank}')

vlm_path = args.vlm_path
feedback = QwenFeedback('cuda',vlm_path,greedy=False)


clip  = 'google/siglip-large-patch16-384'
image_encoder = SiglipVisionModel.from_pretrained(clip)
encoder_image_processor = SiglipImageProcessor.from_pretrained(clip)
image_encoder.to(accelerator.device).to(torch.bfloat16)

import json

if args.dataset == 'geneval':
    with open(args.annotation) as fp:
        metadatas = [json.loads(line) for line in fp]
elif args.dataset == 'custom':
    metadatas = pd.read_csv(args.annotation).to_dict(orient='records')
else:
    metadatas = prepare_dpg_data(csv_file=os.path.join(args.annotation,'dpg_bench.csv'),
                                 prompt_path=os.path.join(args.annotation,'prompts')
                                 )
n = len(metadatas)
n_prompt_per_rank = n // world_size + 1
start = local_rank * n_prompt_per_rank 
all_evaluations = []

acc = 0
total = 0

indices = np.arange(len(metadatas))
rng = np.random.default_rng(420)
shuffled_indices = rng.permutation(indices)
print(shuffled_indices[:100])


reflection_transformer = pipeline.transformer.cpu()
RELFECTION_PROMPT = [
    "Given a user feed back for text-to-image generation, describe how you would fix the image given the feedback ",
    "Feedback: ",
]
for i in range(start,start+n_prompt_per_rank):
    if i >= len(metadatas):
        continue 
    i = shuffled_indices[i]
    outpath = os.path.join(ROOT,f"{i:0>5}") #

    metadata = metadatas[i]
    sample_path = os.path.join(outpath, "samples")

    try:
        os.makedirs(sample_path, exist_ok=True)
    except:
        pass

    os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
        json.dump(metadata, fp)
    #     continue
    if i >= len(metadatas):
        continue 
    
    prompt = metadata['prompt']

    all_feedbacks = []

    print(f"Prompt:{prompt}")
    for j in range(args.num_samples):
        pipeline.to('cuda')
        outpath_file = os.path.join(sample_path, f"{j:05}.{args.fmt}")
        outpath_file_json = outpath_file+'.feedback.json'
        if args.cont and os.path.exists(outpath_file_json):
            load_results = read_json(outpath_file_json)
            if type(load_results) == list:
                feedback_results = load_results[-1]
            else:
                feedback_results = load_results
            if feedback_results['correct_verifier']:
                print(f"{j}: OK (Replayed)")
                acc += 1
                break
            else:
                feedback_payload = dict(
                    path=outpath_file,
                    reason=feedback_results['text_feedback']
                )
                print(f"{j}: {feedback_results['text_feedback']} (Replayed)")
                all_feedbacks.append(feedback_payload)
                all_evaluations.append(feedback_payload)
            continue
        if len(all_feedbacks) == 0:
            imgs = generate_one_image(prompt)
        else:
            N_FEEDBACK = args.n_feedback
            if len(all_feedbacks) <= N_FEEDBACK:
                selected_feedbacks = all_feedbacks
            else:
                selected_feedbacks = subsample_by_unique_reason(all_feedbacks,N_FEEDBACK)
            feedback_imgs = list([ x['path'] for x in selected_feedbacks])
            feedback_texts = list([ x['reason'] for x in selected_feedbacks])
            imgs = sana_pipeline_inference(
                pipeline,
                prompt=prompt,
                height=1024,
                width=1024,
                guidance_scale=4.5,
                num_inference_steps=20,
                do_reflection=not args.no_reflection,
                image_encoder=image_encoder,
                encoder_image_processor=encoder_image_processor,
                n_reflection_tokens_per_img=args.n_reflection_tokens_per_img,
                feedback_imgs = feedback_imgs,
                feedback_texts= feedback_texts,
                complex_human_instruction_feedback=RELFECTION_PROMPT if args.add_prefix_feedback else None,
                drop_feedback=args.drop_feedback
            )
        imgs[0][0].save(outpath_file)
        pipeline.to('cpu')
        feedback_results = feedback.evaluate_image(outpath_file,metadata)

        # 1. model uses `feedback` to generate image
        # 2. `verifier` is used to generate evaluation metrics, saved in variable `actual_results`
        # 3. in actual_results, we add two new fields `correct_verifier` and text_feedback_verifier,
        #    In this context,thet come from `feedback` model in python code, not `verifier` model
        # 4. `verifier`  would be a dummpy object if using custom prompts
        if verifier is not feedback:
            outpath_file_abs = os.path.abspath(outpath_file)
            actual_results = verifier.evaluate_image(outpath_file_abs,metadata)
            actual_results['correct_verifier'] = feedback_results['correct']
            actual_results['text_feedback_verifier'] = feedback_results['text_feedback']
        else:
            actual_results = feedback_results
        actual_results['prompt'] = prompt
        actual_results['gen_idx_gt'] = j
        actual_results['filename'] = outpath_file
        all_evaluations.append(actual_results)
        with open(outpath_file_json,'w') as f:
            f.write(json.dumps(actual_results))
        if feedback_results['correct']:
            print(f"{j}: OK")
            acc += 1
            break
        else:
            feedback_payload = dict(
                path=outpath_file,
                reason=feedback_results['text_feedback']
            )
            print(f"{j}: {feedback_results['text_feedback']}")
            all_feedbacks.append(feedback_payload)
    total += 1
    
annotation_path = os.path.join(ROOT,f'annotations_rank_{local_rank}_of_{world_size}.json')
with open(annotation_path,'w') as f:
    f.write(json.dumps(all_evaluations))
        
accelerator.wait_for_everyone()
