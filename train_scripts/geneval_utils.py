import torch
from transformers import AutoTokenizer
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
import os,re
from transformers import CLIPTextModelWithProjection,CLIPVisionModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast,CLIPImageProcessor
import argparse
import pandas as pd
from argparse import ArgumentParser
import accelerate
# from geneval.evaluation.evaluate_images_distributed import *
from mmdet.apis import inference_detector, init_detector
import open_clip
import mmdet
from PIL import Image, ImageOps
import numpy as np
from clip_benchmark.metrics import zeroshot_classification as zsc
import json

def load_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip()))
    return data


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0

def extract_relevant_object(prompt, position):
    pattern = re.compile(rf"a photo of an? ([\w\s]+) {position} an? ([\w\s]+)")
    match = pattern.search(prompt)
    if match:
        return match.group(2)
    print(f"Could not extract relevant object from prompt: {prompt}")
    return None

class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects,transform):
        self._image = image.convert("RGB")
        bgcolor =  "#999"
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self.transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if '1'== '1':
            image = image.crop(box[:4])
        # if args.save:
        #     base_count = len(os.listdir(args.save))
        #     image.save(os.path.join(args.save, f"cropped_{base_count:05}.png"))
        return (self.transform(image), 0)

def load_models(device,model='gdino'):
    CONFIG_PATH = os.path.join(
            os.path.dirname(mmdet.__file__),
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
    )
    OBJECT_DETECTOR =  "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
    CKPT_PATH = os.path.join("ckpts/geneval-det/", f"{OBJECT_DETECTOR}.pth")
    CKPT_PATH2 = os.path.join("ckpts/geneval-det",f"{OBJECT_DETECTOR}.pth")
    if not os.path.exists(CKPT_PATH):
        CKPT_PATH = CKPT_PATH2
        assert os.path.exists(CKPT_PATH)
    object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=device)

    clip_arch = "ViT-L-14"
    clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open('data/geneval/object_names.txt') as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class SegmentationFeedback:
    
    def __init__(self,device,model='mask2former'):
        object_detector, (clip_model, transform, tokenizer), classnames = load_models(device=device,model=model)
        self.object_detector = object_detector
        self.clip_model= clip_model
        self.transform = transform
        self.tokenizer = tokenizer
        self.classnames = classnames
        self.POSITION_THRESHOLD = 0.1
        self.COLOR_CLASSIFIERS = {}
        self.COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
        self.device = device
        
    def color_classification(self,image, bboxes, classname):
        COLOR_CLASSIFIERS = self.COLOR_CLASSIFIERS
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                self.clip_model, self.tokenizer, self.COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                self.device
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes,self.transform),
            batch_size=16, num_workers=4
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dataloader, self.device)
            return [self.COLORS[index.item()] for index in pred.argmax(1)]

        
    def relative_position(self,obj_a, obj_b):
        """Give position of A relative to B, factoring in object dimensions"""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        #
        revised_offset = np.maximum(np.abs(offset) - self.POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        #
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations


    def evaluate(self,image, objects, metadata):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])[:req['count']]
            if len(found_objects) < req['count']:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if 'color' in req:
                    # Color check
                    colors = self.color_classification(image, found_objects, classname)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found " +
                            f"{colors.count(req['color'])} {req['color']}; and " +
                            ", ".join(f"{colors.count(c)} {c}" for c in self.COLORS if c in colors)
                        )
                if 'position' in req and matched:
                    # Relative position check
                    expected_rel, target_group = req['position']
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found " +
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        # Check for non-expected objects
        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(objects.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)
    
    def extract_reason_info(self,reason,prompt):
        patterns = {
            'expected_count': re.compile(r"expected ([\w\s]+)>=(\d+), found (\d+)"),
            'expected_color': re.compile(r"expected ([\w\s]+) ([\w\s]+)>=(\d+), found (\d+) ([\w\s]+); and (.*)"),
            'expected_position': re.compile(r"expected ([\w\s]+) (left of|right of|above|below) target, found (.*) target"),
            'no_target': re.compile(r"no target for ([\w\s]+) to be ([\w\s]+)"),
            'exclude_count': re.compile(r"expected ([\w\s]+)<(\d+), found (\d+)")
        }

        for key, pattern in patterns.items():
            match = pattern.match(reason)
            if match:
                if key == 'exclude_count':
                    object_class, expected_count, actual_count = match.groups()
                    assert int(expected_count) >0
                    new_reason = f"There should be {expected_count} {object_class}, but only {actual_count} exists in image"
                elif key == 'expected_color':
                    # object_class, color, expected_count, actual_count, other_colors, other_counts = match.groups()
                    # f"There should be {expected_count} {color} {object_class}, but only {actual_count} exists in image; and {other_counts} {other_colors}"
                    raise NotImplementedError()
                elif key == 'expected_position':
                    object_class, position, actual_position = match.groups()
                    #f"{object_class} should be {position} of the {target}, but it is not"
                    target = extract_relevant_object(prompt,position)
                    new_reason = f"{object_class} should be {position} the {target}, but it is {actual_position} the {target}"
                elif key == 'no_target':
                    object_class, position = match.groups()
                    target = extract_relevant_object(prompt,position)
                    new_reason = f"There is no {target} for {object_class} to be {position} of"
                    raise NotImplementedError()
                    # not exist
                elif key == 'expected_count':
                    object_class, expected_count, actual_count = match.groups()
                    if int(actual_count) == 0:
                        new_reason = f"There is no {object_class} in image"
                    else:
                        new_reason = f"There should be {expected_count} {object_class}, but only {actual_count} exists in image"
                        #breakpoint()
                return key, match.groups(),new_reason
        print(f"Could not extract info from reason: {reason}")
        return None, None
        
    def evaluate_image(self,filepath, metadata):
        object_detector = self.object_detector
        THRESHOLD = 0.3 #float(args.options.get('threshold', 0.3))
        COUNTING_THRESHOLD = 0.9 #float(args.options.get('counting_threshold', 0.9))
        MAX_OBJECTS = 16 #int(args.options.get('max_objects', 16))
        NMS_THRESHOLD = 1.0 #float(args.options.get('max_overlap', 1.0))
        POSITION_THRESHOLD = 0.1 #float(args.options.get('position_threshold', 0.1))

        result = inference_detector(object_detector, filepath)
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(Image.open(filepath))
        detected = {}
        # Determine bounding boxes to keep
        confidence_threshold = THRESHOLD if metadata['tag'] != "counting" else COUNTING_THRESHOLD
        for index, classname in enumerate(self.classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold] # Threshold
            ordering = ordering[:MAX_OBJECTS].tolist() # Limit number of detected objects per class
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                ordering = [
                    obj for obj in ordering
                    if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
                ]
            if not detected[classname]:
                del detected[classname]
        # Evaluate
        is_correct, reason = self.evaluate(image, detected, metadata)
        if reason:
            _,_,text_feedback = self.extract_reason_info(reason,metadata['prompt'])
        else:
            text_feedback = ''
        return {
            'filename': filepath,
            'tag': metadata['tag'],
            'prompt': metadata['prompt'],
            'text_feedback':text_feedback, # required
            'correct': is_correct, # required
            'reason': reason,
            'metadata': json.dumps(metadata),
            'details': json.dumps({
                key: [box.tolist() for box, _ in value]
                for key, value in detected.items()
            })
        }

import torchvision.ops as ops        
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
       
class QwenFeedback:
    
    def __init__(self,device,path="Qwen/Qwen2.5-VL-3B-Instruct",greedy=False):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        self.device = device
        self.greedy = greedy
    def build_message(self,filepath,prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": filepath,
                    },
                    {"type": "text", "text": f"Please evaluate this generated image based on the following prompt: {prompt}. Focus on text alignment and compositionality."},
                ],
            }
        ]
        return messages
    def evaluate_image(self,filepath, metadata):
        messages = self.build_message(filepath,metadata['prompt'])
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        if self.greedy:
            generated_ids = self.model.generate(**inputs, max_new_tokens=128,do_sample=False)
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        res = output_text[0]
        correct = 'is correct' in res
        if not correct:
            reason = res.split('.')[-1].strip()
        else:
            reason = ''
        return dict(
            correct=correct,
            text_feedback=reason
        )
  
def save_jsonl(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
