<h1 align="center">
- AI captcha solver adapter
</h1>

## Requirements
> Python3.3 - python3.10

~~Python 3.10 is not supported yet because 
[onnxruntime](https://pypi.org/project/onnxruntime/) 
is not supporting **python3.10**~~

#### UPDATE: Python3.10 is supported

## Installation

```
pip install captcha-solve-adapter   
```

```python
from captcha_solve_adapter import CaptchaSolver

solver = CaptchaSolver(
    logging=False, # if need to print the log
    img_width=300, # img_width
    img_height=40, # img height
    max_length=10, # max captcha length
    characters=['a','b','c','d'], # captcha characters used in training model
    model_fname='Path/to/the/model.onnx'    
)  # this login will create captcha
def solve_captcha(url: str):
    result, accur = solver.solve(url=url)
    return accur
def solve_from_bytes(b: bytes):
    result, accur = solver.solve(bytes_data=b)
    return accur
def solve_frmo_file(file: str):
    with open(file, 'rb') as f:
        b = f.read()
    return solve_from_bytes(b)
```
