import os
import erniebot as eb

from ..presets import *
from ..utils import *

from .base_model import BaseLLMModel


class ERNIE_Client(BaseLLMModel):
    def __init__(self, model_name, api_key, secret_key, api_type: str = "aistudio", access_token: str = None) -> None:
        super().__init__(model_name=model_name)
        self.auth_config = {"api_type": api_type, "ak": api_key, "sk": secret_key}
        if access_token:
            self.auth_config["access_token"] = access_token

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

        if self.model_name == "chat_file":
            response = eb.ChatFile.create(_config_=self.auth_config, **data, stream=True)
        else:
            response = eb.ChatCompletion.create(
                _config_=self.auth_config, model=self.model_name, **data, stream=True
            )
        for result in response:
            yield result.get_result()


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

        if self.model_name == "chat_file":
            return eb.ChatFile.create(_config_=self.auth_config, **data, stream=False).get_result()
        else:
            return eb.ChatCompletion.create(
                _config_=self.auth_config, model=self.model_name, **data, stream=False
            ).get_result()
