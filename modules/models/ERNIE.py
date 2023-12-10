import os
import qianfan

from ..presets import *
from ..utils import *

from .base_model import BaseLLMModel


class ERNIE_Client(BaseLLMModel):
    def __init__(self, model_name, api_key, secret_key) -> None:
        super().__init__(model_name=model_name)
        self.chat_completion = qianfan.ChatCompletion(ak=api_key, sk=secret_key)

    def get_answer_stream_iter(self):
        system_prompt = self.system_prompt
        history = self.history
        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        # 去除history中 history的role为system的
        history = [i for i in history if i["role"] != "system"]

        data = {
            "messages": history,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }

        response = self.chat_completion.do(model=self.model_name, **data, stream=True)
        partial_text = ""
        for result in response:
            partial_text += result['result']
            yield partial_text

    def get_answer_at_once(self):
        system_prompt = self.system_prompt
        history = self.history
        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        # 去除history中 history的role为system的
        history = [i for i in history if i["role"] != "system"]

        data = {
            "messages": history,
            "top_p": self.top_p,
            "temperature": self.temperature,
        }

        return self.chat_completion.do(model=self.model_name, **data)['result']
