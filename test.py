import captcha_solver
s = captcha_solver.CaptchaSolver()
with open(r'D:\Projects\Python\CaptchaSolver123\images\test\2dd8.png', 'rb') as f:
    b_data = f.read()
print(s.solve(bytes_data=b_data))


