
def test():
    torch.manual_seed(0)
    model = unet_residual()
    print('model type: ', model.__class__.__name__)
    num_params = sum([p.data.nelement() for p in model.parameters()])
    print('number of trainable parameters: ', num_params)
    x = torch.randn(8, 1, 4, 128, 128)
    x[:2] = 0.3
    y = model(x)
    print(x.size(), y.size(), x.mean(), y.mean())

if __name__ == '__main__':
    test()
