from model import convnet, mlp, resnet

def make_model(args, n_classes, n_channels, device, img_size=32):
    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])

    if args.model == "convnet":
        model = convnet.LeNet5(n_classes, n_channels, img_size, conv_hidden_size, dense_hidden_size, device)
    elif args.model == "mlp":
        model = mlp.MLP(n_classes, dense_hidden_size, device)
    elif args.model == "resnet":
        model = resnet.resnet20().to(device)
    else:
        raise NotImplementedError

    return model