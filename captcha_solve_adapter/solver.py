from typing import Any

import concurrent.futures
import os.path
import time
import aiohttp, asyncio

import onnxruntime as onr
import numpy as np
import requests
import cv2
import threading
from requests.exceptions import ProxyError


lock = threading.Lock()
logging_lock: "threading.Lock|None" = None  # you can use some existing lock for logging
class CaptchaSolver:
    """
    Captcha solver adapter handling
    Fast examples:
1) solving captcha from url
>>> from captcha_solve_adapter import CaptchaSolver
>>> import random
>>> solver = CaptchaSolver(logging=True)
>>> captcha_response, accuracy = solver.solve(url='http://link-to-captcha')
>>> async def async_way():
... await solver.solve_async(url=f"http://link-to-captcha", ) # session:aiohttp:ClientSession)
2) if you have an image in bytes:
>>> solver.solve(bytes_data=requests.get(f"http://link-to-captcha").content)
    """
    img_width: int
    TOTAL_COUNT = 0
    FAIL_COUNT = 1
    TOTAL_TIME = 0
    def __init__(
            self,
            logging=False,
            model_fname=os.path.dirname(__file__) + "/model.onnx",
            max_length=7,
            characters=['q', 'd', 'n', 'u', 'f', 'g', 'h', 'x', 'p', 'e', 'z', 'b', 't', 'm', 'r', 'a', 'c', 'y', 'j', '8',
                  '3', '4', '7', '9', '2', '6'],
            img_width=140,
            img_height = 36
    ):
        self.max_length = max_length
        self.characters = characters
        self.img_width = img_width
        self.img_height = img_height
        self.logging = logging
        self.Model = onr.InferenceSession(model_fname)
        self.ModelName = self.Model.get_inputs()[0].name
    def solve(self, url=None, bytes_data=None, session=None) -> 'str,float':
        """Solves VK captcha
        :param bytes_data: Raw image data
        :type bytes_data: bytes
        :param url: url of the captcha ( or pass bytes_data )
        :type url: str
        :param session: requests.Session object or None
        :return Tuple[answer:str, accuracy:float ( Range=[0,1]) ]
        """
        if self.logging:
            with logging_lock:
                print(f"Solving captcha {url}")
        if url is not None:
            for _ in range(4):
                try:
                    bytes_data = (session or requests).get(url, headers={"Content-language": "en"}).content
                    if bytes_data is None:
                        raise ProxyError("Can not download data, probably proxy error")
                    break
                except:
                    if _ == 3: raise
                    time.sleep(0.5)
        answer, accuracy = self._solve_task(bytes_data)
        with lock: CaptchaSolver.TOTAL_COUNT += 1
        return answer, accuracy

    @property
    def argv_solve_time(self):
        """Argv solve time in seconds per one captcha.
        Start returning value after first solve ( solve_async) call"""
        with lock:
            return CaptchaSolver.TOTAL_TIME / (CaptchaSolver.TOTAL_COUNT or 1)  # zero division error capturing
    @property
    def _async_runner(self):
        if not hasattr(CaptchaSolver, "_runner"):
            CaptchaSolver._runner = concurrent.futures.ThreadPoolExecutor(
                max_workers=5
            )
        return CaptchaSolver._runner
    async def solve_async(self, url=None, bytes_data=None, session=None) -> 'str,float':
        """Solves VK captcha async
        :param bytes_data: Raw image data
        :type bytes_data: byte
        :param url: url of the captcha ( or pass bytes_data )
        :type url: str
        :param session: aiohttp.ClientSession session to download captcha
        :type session: aiohttp.ClientSession
        :return answer:str, accuracy:float ( Range=[0,1])
        """
        if self.logging: print(f"Solving captcha {url}")
        if url is not None:
            for _ in range(4):
                try:
                    if session is None:
                        async with aiohttp.ClientSession(headers={"Content-language": "en"}) as session_m, \
                                session_m.get(url) as resp:
                            bytes_data = await resp.content.read()
                    else:
                        async with session.get(url) as resp:
                            bytes_data = await resp.content.read()
                    if bytes_data is None: raise ProxyError("Can not download captcha - probably proxy error")
                    break
                except Exception:
                    if _ == 3: raise
                    await asyncio.sleep(0.5)
        if self.logging: t = time.time()
        #  running in background async
        res = asyncio.get_event_loop().run_in_executor(self._async_runner, self._solve_task, bytes_data)
        completed, _ = await asyncio.wait((res,))
        #  getting result
        answer, accuracy = next(iter(completed)).result()
        with lock: CaptchaSolver.TOTAL_COUNT += 1
        return answer, accuracy
    def _solve_task(self, data_bytes: bytes):
        t = time.time()
        if data_bytes[:3] == b'GIF':
            # Gif is not supported by cv2 by itself - we have to use imageio as a spacer
            import imageio.v3 as iio
            gif = iio.imread(data_bytes, index=None, format_hint=".gif")
            img = cv2.cvtColor(gif[0], cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imdecode(np.asarray(bytearray(data_bytes), dtype=np.uint8), -1)

        if len(img.shape) > 2 and img.shape[2] == 4:
            # If image is 4 channels (we need 3) then
            # convert from RGBA2RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img: "np.ndarray" = img.astype(np.float32) / 255.
        if img.shape != (self.img_height, self.img_width, 3):
            cv2.resize(img, (self.img_width, self.img_height))
        img = img.transpose([1, 0, 2])
        #  Creating tensor ( adding 4d dimension )
        img = np.array([img])
        # !!!HERE MAGIC COMES!!!!
        result_tensor = self.Model.run(None, {self.ModelName: img})[0]
        # decoding output
        answer, accuracy = self.get_result(result_tensor)

        delta = time.time() - t
        with lock:
            CaptchaSolver.TOTAL_TIME += delta
        if self.logging:
            with logging_lock:
                print(f"Solved captcha = {answer} ({accuracy:.2%} {time.time() - t:.3}sec.)")

        return answer, accuracy

    def get_result(self, pred):
        """CTC decoder of the output tensor
        https://distill.pub/2017/ctc/
        https://en.wikipedia.org/wiki/Connectionist_temporal_classification
        :return string, float
        """
        accuracy = 1
        last = None
        ans = []
        # pred - 3d tensor, we need 2d array - first element
        for item in pred[0]:
            # get index of element with max accuracy
            char_ind = item.argmax()
            # ignore duplicates and special characters
            if char_ind != last and char_ind != 0 and char_ind != len(self.characters) + 1:
                # this element is a character - append it to answer
                ans.append(self.characters[char_ind - 1])
                # Get accuracy for current character and
                # multiply global accuracy by it
                accuracy *= item[char_ind]
            last = char_ind

        answ = "".join(ans)[:self.max_length]
        return answ, accuracy
