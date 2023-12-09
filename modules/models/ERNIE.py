import erniebot as eb

from ..presets import *
from ..utils import *

from .base_model import BaseLLMModel


class ERNIE_Client(BaseLLMModel):
    def __init__(self, model_name, api_key, secret_key, api_type: str = "aistudio", access_token: str = None) -> None:
        super().__init__(model_name=model_name)
        self.auth_config = {
            "api_type": api_type,
        }
        self.auth_config["ak"] = api_key
        self.auth_config["sk"] = secret_key
        if access_token:
            self.auth_config["access_token"] = access_token
        self.eb = eb.ChatCompletion

        if self.model_name == "ERNIE-Bot-turbo":
            self.ERNIE_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token="
        elif self.model_name == "ERNIE-Bot":
            self.ERNIE_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token="
        elif self.model_name == "ERNIE-Bot-4":
            self.ERNIE_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token="

    def get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token?client_id=" + self.api_key + "&client_secret=" + self.api_secret + "&grant_type=client_credentials"

        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        return response.json()["access_token"]
    def get_answer_stream_iter(self):
        url = self.ERNIE_url + self.get_access_token()
        system_prompt = self.system_prompt
        history = self.history
        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        # 去除history中 history的role为system的
        history = [i for i in history if i["role"] != "system"]

        payload = json.dumps({
            "messages":history,
            "stream": True
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload, stream=True)

        if response.status_code == 200:
            partial_text = ""
            for line in response.iter_lines():
                if len(line) == 0:
                    continue
                line = json.loads(line[5:])
                partial_text += line['result']
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG


    def get_answer_at_once(self):
        url = self.ERNIE_url + self.get_access_token()
        system_prompt = self.system_prompt
        history = self.history
        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        # 去除history中 history的role为system的
        history = [i for i in history if i["role"] != "system"]

        payload = json.dumps({
            "messages": history,
            "stream": True
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload, stream=True)

        if response.status_code == 200:

            return str(response.json()["result"]),len(response.json()["result"])
        else:
            return "获取资源错误", 0


