import torch

# 1, 2, 3이 들어 있는 텐서를 만듭니다.
a = torch.tensor([1, 2, 3])
# 4, 5, 6이 들어 있는 텐서를 만듭니다.
b = torch.tensor([4, 5, 6])
# 두 텐서의 합을 구합니다.
c = a + b

# 텐서를 출력합니다.
print(c)