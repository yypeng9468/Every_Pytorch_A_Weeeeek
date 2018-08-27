# torch.cuda 主要是用来设置 cuda 的参数的，通过持续跟踪目前选中的 GPU，
# 默认情况下所有的 cuda tensors 都在此块GPU上生成，可以通过改变
# torch.cuda.device 来改变 GPU id

# 默认情况下 cross-GPU 操作是不被允许的，除了 copy_ 和 to() 以及 cuda()
import torch
cuda = torch.device('cuda') # 默认的 CUDA 机器
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2') # GPU 2

x = torch.tensor([1., 2.], device=cuda0)
y = torch.tensor([1., 2.]).cuda()

with torch.cuda.device(1):
    # 在 GPU 1 上分配一个 tensor
    a = torch.tensor([1., 2.], device=cuda)

    # 将一个 tensor 从 CPU 迁移到 GPU 上
    b = torch.tensor([1., 2.]).cuda()
    # 或者用 to 可以达到同样的效果
    b2 = torch.tensor([1., 2.]).to(device=cuda)
    # 上面的 a, b, b2 都位于 GPU 1 上

    c = a + b
    # c 位于 GPU 1
    z = x + y
    # z 位于 GPU 0

    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)
    f = torch.randn(2).cuda(cuda2)


## Device agnostic code
import argparse
parser = argparse.ArgumentParser(description="Pytorch Example")
parser.add_argument('--disable-cuda', action='store_true', help="Disable CUDA")
args = parser.parse_args()
args.device = None

if not arg.disable_cuda and not torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

x = torch.empty((8,42), device=args.device)
net = Network().to(device=arg.device)

