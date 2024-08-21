from typing import List, Optional
from cog import BasePredictor, Input
from transformers import AutoTokenizer
import torch
 
import os
from vllm import LLM, SamplingParams

os.environ['HF_TOKEN'] = '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'


sampling_params = SamplingParams(
    temperature=0.3,
    max_tokens=1024,
)


class Predictor(BasePredictor):
    def setup(self):
        self.llm = LLM(model="SM0rc/mathleaks-grapher-merged", max_model_len=42592)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.outputs = []
        
    def predict(
        self,
        descriptions: str = Input(description=f"Text prompt to send to the model.")
        ) -> List[str]:

        chats = [self.tokenizer.apply_chat_template([
            { "content": "Convert the description to jsxgpre code", "role": "system" },
            { "content": description, "role": "user" }, 
            { "content": "", "role": "assistant" } 
        ], tokenize=False).replace('assistant<|end_header_id|>\n\n<|eot_id|>', 'assistant<|end_header_id|>\n\n') for description in descriptions]


        outputs = self.llm.generate(
            chats,
            sampling_params,
        )

        for output in outputs:
            self.outputs.append(output.outputs[0].text.replace('\\n', '\n'))

        return self.outputs
