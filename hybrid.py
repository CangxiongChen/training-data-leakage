# hybrid method
# reuse model dictionary created in cop_attack.py that contains info
# on grads of each weights, values of weights etc

import numpy as np
from circulant_rep import circ_rep_grad, \
    circulant_rep_single_channel, circulant_rep
from targets import cnn3_c3, cnn3_c1, cnn3_c2, cnn3_c4, cnn3_c11, cnn3_c21, cnn3_c31, cnn3_c41,\
                    LeNet, cnn2_c1, cnn2_c2, cnn2_c4, cnn2_c5, cnn2_c11, cnn2_c21, cnn2_c41, cnn2_c0,\
                    cnn4_c1, cnn2_c5, cnn2_c6, LeNet2, cnn3_c5, cnn4_c2, cnn4_c3


def total_variation(x):
    """Anisotropic TV."""
    import torch
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def der_of_activation(input, act_type):
    """
    derivative of given activation function
    input:  numpy array, shape (num of elements,)
    return: square diagonal matrix of shape n by n, where n: num of elements in input
    """
    import numpy as np

    if act_type == 'leakyrelu':

        def der_leakyrelu(x, alpha=0.01):
            if x > 0:
                result = 1.0
            elif x < 0:
                result = alpha
            elif x == 0:
                print("The derivative does not exist at {}")
            return result
        der = np.array([der_leakyrelu(x) for x in input.flatten()])

    elif act_type == 'sigmoid':
        der = 1.0 / (2.0 + np.exp(input) + np.exp(-input))

    elif act_type == 'tanh':
        der = 1.0/np.cosh(input)**2

    elif act_type == 'identity':
        der = np.ones(shape=input.shape)

    elif act_type == 'softplus':
        beta = 1
        der = 1.0 / (1.0 + np.exp(-beta*input))

    elif act_type == 'identity':
        der = np.ones(input.shape)

    else:
        raise ValueError("Activation not included!")

    der = np.diag(der.reshape((input.size,)))

    return der


def inv_of_activation(input, act_type):
    """
    inverse of given activation function
    input:  numpy array, shape (num of elements,)
    return: numpy array of same shape as the input
    """
    import numpy as np
    if act_type == 'leakyrelu':
        def inv_leakyrelu(x, alpha=0.01):
            if x >= 0:
                result = x
            elif x < 0:
                result = x * 1.0 / alpha
            return result

        inv = np.array([inv_leakyrelu(x) for x in input.flatten()])

    elif act_type == 'sigmoid':
        res = np.multiply(input, 1.0/(1.0 - input))
        inv = np.log(res)

    elif act_type == 'tanh':
        input_clipped = np.clip(input,-0.9999,0.9999)
        inv = np.arctanh(input_clipped)
    elif act_type == 'identity':
        inv = input

    elif act_type == 'softplus':
        beta = 1.0
        inv = (1.0/beta)*np.log(np.exp(beta*input) - 1.0)

    else:
        raise ValueError("Activation not included!")

    inv = inv.reshape(input.shape)
    return inv


def create_model_info(model, input, label, loss_fn, epochs=1):
    """
    initialise the weights, compute the gradients of the cost function
    from one pass of the training data

    return: dictionary of all the info
    """
    import torch
    from torch import nn
    from torch.optim import SGD
    if epochs == 1:
        # print(" --- Untrained model, using randomised weights --- ")
        output = model(input)[0]
        loss = loss_fn(output,label)
        loss.backward()
    elif epochs > 1:
        # print(" --- Pretrained model, using latest weights --- ")
        optimiser = SGD(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            optimiser.zero_grad()
            output = model(input)[0]
            loss = loss_fn(output,label)
            loss.backward()
            optimiser.step()

    modules = model.module_list
    acts = model.activation_list

    model_info = {}
    conv_count = 0
    x = input
    for i, (layer, act) in enumerate(zip(modules,acts)):
        if isinstance(layer, torch.nn.Linear):
            x = torch.flatten(x,1)
            padding = 'none'
            stride = 0
        elif isinstance(layer, torch.nn.Conv2d):
            conv_count += 1
            padding = layer.padding[0]
            stride = layer.stride[0]

        if not isinstance(layer.bias,type(None)):
            bias = layer.bias.detach().cpu().numpy()
            grad_bias = layer.bias.grad.cpu().numpy()
        elif isinstance(layer.bias,type(None)):
            bias = 0
            grad_bias = 0

        if isinstance(act, nn.LeakyReLU):
            act_type = 'leakyrelu'
        elif isinstance(act, nn.Sigmoid):
            act_type = 'sigmoid'
        elif isinstance(act, nn.Tanh):
            act_type = 'tanh'
        elif isinstance(act, nn.Identity):
            act_type = 'identity'
        elif isinstance(act, nn.Softplus):
            act_type = 'softplus'
        else:
            raise ValueError("Specified activation not included!")
        layer_out = layer(x)
        act_out = act(layer_out)
        layer_info = {'weight': layer.weight.detach().clone().cpu().numpy(),
                      'grad_weight': layer.weight.grad.detach().clone().cpu().numpy(),
                      'bias': bias,
                      'grad_bias': grad_bias,
                      'input_shape': x.shape,
                      'z_shape': layer_out.shape,
                      'padding': padding,
                      'layer_in': torch.flatten(x,1).detach().clone().cpu().numpy(),
                      'input_to_layer': x.detach().clone(),
                      'act_out': torch.flatten(act_out,1).detach().clone().cpu().numpy(),
                      'stride': stride,
                      'act_type': act_type,
                      'latent_out': torch.flatten(layer_out,1).detach().clone().cpu().numpy()}

        model_info[str(i)] = layer_info
        x = act_out

    model_info['target_label'] = label
    return model_info


def get_U_V_conv(in_shape, weight_circ, z_shape, Dz, Dw, inv_act, stride, padding):
    """
    adapted from https://github.com/JunyiZhu-AI/R-GAP/blob/b842eca0e85029784a793639acd6a4047e738af1/recursive_attack.py#L83
    """
    import time
    from scipy import sparse

    start_time = time.time()

    inv_act = inv_act.reshape((-1,1))

    W = weight_circ
    Dz_circ = circ_rep_grad(weight_shape=Dw.shape, der_z=Dz.reshape(z_shape), x_shape=in_shape, stride=stride, padding=padding)
    print("Shape of W, K for the current layer is {},{}\n".format(W.shape, Dz_circ.shape))
    print("Shape of the inverse of activation is {}".format(inv_act.shape))

    # transform Dw to be sparse before concatenations
    Dw_flat = Dw.reshape((-1,1))
    print("Shape of Dw is {}".format(Dw.shape))
    U = sparse.vstack([W, Dz_circ], format="csr")
    V = sparse.vstack([inv_act, Dw_flat], format="csr")

    print("Shape of U is {}".format(U.shape))
    print("Shape of V is {}".format(V.shape))
    print(" --- computing U and V in {} seconds --- ".format(time.time() - start_time))

    return U, V


def generate_coordinates(x_shape, kernel, stride, padding):
    """
    From https://github.com/JunyiZhu-AI/R-GAP/.
    """
    assert len(x_shape) == 4
    assert len(kernel.shape) == 4
    assert x_shape[1] == kernel.shape[1]
    k_i, k_j = kernel.shape[-2:]
    x_i, x_j = np.array(x_shape[-2:])+2*padding
    y_i, y_j = (x_i-k_i)//stride+1, (x_j-k_j)//stride+1
    kernel = kernel.reshape(kernel.shape[0], -1)
    circulant_w = []
    for f in range(len(kernel)):
        circulant_row = []
        for u in range(len(kernel[f])):
            c = u // (k_i*k_j)
            h = (u - c*k_i*k_j) // k_j
            w = u - c*k_i*k_j - h*k_j
            rows = np.array(range(0, x_i-k_i+1, stride)) + h
            cols = np.array(range(0, x_j-k_j+1, stride)) + w
            circulant_unit = []
            for row in range(len(rows)):
                for col in range(len(cols)):
                    circulant_unit.append([f*y_i*y_j+row*y_j+col, c*x_i*x_j+rows[row]*x_j+cols[col]])
            circulant_row.append(circulant_unit)
        circulant_w.append(circulant_row)
    return np.array(circulant_w), x_shape[1]*x_i*x_j, kernel.shape[0]*y_i*y_j


def aggregate_g(k, x_len, coors):
    """
    From https://github.com/JunyiZhu-AI/R-GAP/.
    """
    k = k.squeeze()
    A_mat = []
    for coor in coors:
        A_row = []
        for c in coor:
            A_unit = np.zeros(shape=x_len, dtype=np.float32)
            for i in c:
                assert A_unit[i[1]] == 0
                A_unit[i[1]] = k[i[0]]
            A_row.append(A_unit)
        A_mat.append(A_row)
    A_mat = np.array(A_mat)
    return A_mat.reshape(-1, A_mat.shape[-1])


def circulant_w(x_len, kernel, coors, y_len):
    weights = np.zeros([y_len, x_len], dtype=np.float32)
    kernel = kernel.reshape(kernel.shape[0], -1)
    for coor, f in list(zip(coors, kernel)):
        for c, v in list(zip(coor, f)):
            for h, w in c:
                assert weights[h, w] == 0
                weights[h, w] = v
    return weights


def padding_constraints(in_shape, padding):
    """
    From https://github.com/JunyiZhu-AI/R-GAP/.
    """
    toremain = peeling(in_shape, padding)
    P = []
    for i in range(toremain.size):
        if not toremain[i]:
            P_row = np.zeros(toremain.size, dtype=np.float32)
            P_row[i] = 1
            P.append(P_row)
    return np.array(P)


def peeling(in_shape, padding):
    """
    From https://github.com/JunyiZhu-AI/R-GAP/.
    """
    if padding == 0:
        # return np.ones(shape=in_shape, dtype=bool).squeeze()
        return np.ones(shape=(in_shape.numel(),), dtype=bool)
    h, w = np.array(in_shape[-2:]) + 2*padding
    toremain = np.ones(h*w*in_shape[1], dtype=bool)
    if padding:
        for c in range(in_shape[1]):
            for row in range(h):
                for col in range(w):
                    if col < padding or w-col <= padding or row < padding or h-row <= padding:
                        i = c*h*w + row*w + col
                        assert toremain[i]
                        toremain[i] = False
    return toremain


def get_U_V_conv_v2(in_shape, weight, der_z, g, inv_act, stride, padding):
    """
    adapted from https://github.com/JunyiZhu-AI/R-GAP/blob/b842eca0e85029784a793639acd6a4047e738af1/recursive_attack.py#L83
    Using R-GAP's implementation to get padding.
    """
    import time
    from scipy import sparse

    coors, x_len, y_len = generate_coordinates(x_shape=in_shape, kernel=weight, stride=stride, padding=padding)
    K = aggregate_g(k=der_z, x_len=x_len, coors=coors)
    print("Shape of K {}\n".format(K.shape))
    W = circulant_w(x_len=x_len, kernel=weight, coors=coors, y_len=y_len)
    print("Shape of W {}\n".format(W.shape))
    P = padding_constraints(in_shape=in_shape, padding=padding)
    p = np.zeros(shape=P.shape[0], dtype=np.float32)
    out = inv_act.reshape(-1)
    if np.any(P):
        a = np.concatenate((W, K, P), axis=0)
        b = np.concatenate((out, g.reshape(-1), p), axis=0)
    else:
        a = np.concatenate((W, K), axis=0)
        b = np.concatenate((out, g.reshape(-1)), axis=0)
    print("Shape of U,V {},{}\n".format(a.shape,b.shape))
    rank_comp_start_time = time.time()
    from numpy.linalg import matrix_rank
    rank_u = matrix_rank(a,tol=1e-10)
    print(" --- rank of W, K,  U, [U|V] with tolerance {rW},{rK},{rU},{rUV},{tol} --- ".format(
        rW=matrix_rank(W,tol=1e-10),
        rK=matrix_rank(K,tol=1e-10),
        rU=rank_u,
        rUV=matrix_rank(np.hstack((a,b.reshape((-1,1)))), tol=1e-10),
        tol=1e-10))
    rank_deficiency = rank_u - a.shape[1]
    rank_comp_duration = time.time() - rank_comp_start_time
    print(" --- Rank computations take {} seconds --- ".format(rank_comp_duration))
    a_sparse = sparse.csr_matrix(a)
    b_sparse = sparse.csr_matrix(b.reshape(-1,1))
    return a_sparse, b_sparse, rank_deficiency


def fully_connected_invert(grad_weight, grad_bias, weight):
    import numpy as np
    """
    Reconstruct the input to a fully connected layer using the gradients relation from the chain rule.
    reparameterise: reparameterise the output of the reconstruction based on activation from the preceding
    layer 
    grad_weight,grad_act, grad_v: gradients wrt the weights, the activations and the derivative of the activation
    w.r.t. its input
    input shapes:
    grad_weights: n^2
    grad_v: n
    grad_act: n
    All need to be numpy arrays.
    return: reconstructed input to the linear layer, derivative of x
    """
    print("Reconstructing the fully connected layer!")
    # using the relation valid for fully connected layer:
    grad_z = grad_bias
    x = np.zeros(grad_weight.shape[1])
    # compute the derivative of J wrt x, assuming tanh activation
    der_x = np.matmul(grad_z, weight)
    # find the first nonzero element
    grad_z = grad_z.reshape((grad_z.size,))
    indx = grad_z.nonzero()[0][0]
    v = grad_z[indx]

    for j in range(len(x)):
        x[j] = grad_weight[indx,j]*1.0 / v
    x_reshape = x.reshape((1,x.size))
    print("Reconstruction for the fully connected layer is done!")
    return x_reshape, der_x


class MyTransform:
    """
    First recenter the image by image = image - 0.5 (assuming image lies in the range [0,1]; Then
    apply transforms.ToTensor
    """

    def __call__(self, image):
        from torchvision.transforms import ToTensor
        to_tensor = ToTensor()
        image = to_tensor(image)
        image = image - 0.5
        return image


def my_transpose(input_image):
    """helper function for plotting"""
    img = input_image
    out_img = img.transpose((1,2,0))
    return out_img


def plot_compare(orig_img, rgap_recon, dlg_recon, hybrid_recon, geiping_recon, root_directory, test_name):
    import matplotlib.pyplot as plt
    import os
    output_directory = root_directory
    output_filename = os.path.join(output_directory, "comparisons_with_scores_{}.png".format(test_name))

    target_image = np.copy(orig_img.cpu().detach().numpy())
    rgap_recon_ = np.copy(rgap_recon)
    dlg_recon_ = np.copy(dlg_recon)
    hybrid_recon_ = np.copy(hybrid_recon)
    geiping_recon_ = np.copy(geiping_recon)

    target_imag_trans = my_transpose(target_image[0,:,:,:]).squeeze()
    rgap_recon_trans = my_transpose(rgap_recon_[0,:,:,:]).squeeze()
    dlg_recon_trans = my_transpose(dlg_recon_[0,:,:,:]).squeeze()
    hybrid_recon_trans = my_transpose(hybrid_recon_[0,:,:,:]).squeeze()
    geiping_recon_trans = my_transpose(geiping_recon_[0,:,:,:]).squeeze()

    target_imag_trans += 0.5
    rgap_recon_trans += 0.5
    dlg_recon_trans += 0.5
    hybrid_recon_trans += 0.5
    geiping_recon_trans += 0.5

    # save the images as npy files for further plotting
    np.save(os.path.join(output_directory, "{}_target_img.npy".format(test_name)), target_imag_trans)
    np.save(os.path.join(output_directory, "{}_rgap_img.npy".format(test_name)), rgap_recon_trans)
    np.save(os.path.join(output_directory, "{}_dlg_img.npy".format(test_name)), dlg_recon_trans)
    np.save(os.path.join(output_directory, "{}_cosine_img.npy".format(test_name)), geiping_recon_trans)
    np.save(os.path.join(output_directory, "{}_hybrid_img.npy".format(test_name)), hybrid_recon_trans)
    print("Image files saved to {}!\n".format(os.path.join(output_directory,
                                             "{}_target_img.npy".format(test_name)), target_imag_trans))

    mse_dlg = np.mean(np.square(target_imag_trans - dlg_recon_trans))
    mse_rgap = np.mean(np.square(target_imag_trans - rgap_recon_trans))
    mse_hyb = np.mean(np.square(target_imag_trans - hybrid_recon_trans))
    mse_geiping = np.mean(np.square(target_imag_trans - geiping_recon_trans))

    psnr_dlg = 20.0*np.log10(255.0) - 10.0*np.log10(mse_dlg)
    psnr_rgap = 20.0*np.log10(255.0) - 10.0*np.log10(mse_rgap)
    psnr_hyb = 20.0*np.log10(255.0) - 10.0*np.log10(mse_hyb)
    psnr_geiping = 20.0*np.log10(255.0) - 10.0*np.log10(mse_geiping)


    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,figsize=(10,10))
    ax1.set_title("Target", fontsize=20)
    ax1.axis('off')
    ax1.imshow(target_imag_trans,vmin=0.0,vmax=1.0)

    ax2.set_title("R-GAP", fontsize=18)
    ax2.axis('off')
    ax2.imshow(rgap_recon_trans,vmin=0.0,vmax=1.0)

    ax3.set_title("DLG", fontsize=18)
    ax3.axis('off')
    ax3.imshow(dlg_recon_trans, vmin=0.0,vmax=1.0)

    ax4.set_title("Cosine", fontsize=20)
    ax4.axis('off')
    ax4.imshow(geiping_recon_trans, vmin=0.0,vmax=1.0)

    ax5.set_title("Hybrid", fontsize=20)
    ax5.axis('off')
    ax5.imshow(hybrid_recon_trans, vmin=0.0,vmax=1.0)

    fig.suptitle(f"| R-GAP: {mse_rgap:2.4f}, {psnr_rgap:2.4f} "
                 f"| DLG: {mse_dlg:2.4f}, {psnr_dlg:2.4f} "
                 f"| Cosine: {mse_geiping:2.4f},{psnr_geiping:2.4f} "
                 f"| HYB: {mse_hyb:2.4f}, {psnr_hyb:2.4f} ",
                 fontsize=20, wrap=True)

    plt.savefig(output_filename)
    print("Plots saved to {}!\n".format(output_filename))


def test_on_CIFAR10(data_path, index):
    import torch
    import torchvision
    train_ind = [index]
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=MyTransform())
    trainsubset = torch.utils.data.Subset(trainset, train_ind)
    trainloader = torch.utils.data.DataLoader(trainsubset, batch_size=1, shuffle=True)

    return trainloader


def make_w_circulant(modules, model_info):
    """
    for each conv2d layer, transform the weight to be its circulant form; for a linear layer, keep w the same
    """
    import torch
    weight_list = []
    for i, layer in enumerate(modules):
        weight = model_info[str(i)]['weight']
        if isinstance(layer,torch.nn.Conv2d):
            stride = model_info[str(i)]['stride']
            weight = circulant_rep(weight=weight, input_h=model_info[str(i)]['input_shape'][2],
                                      padding=model_info[str(i)]['padding'], stride=stride,
                                      dilation=1, bias='none')
            weight = weight.toarray()
        weight_list.append(weight)

    return weight_list


def conv_layer_der_x(der_x_next, weight, z, activation):
    """
    Compute gradient of J wrt layer output x for the current layer.
    """
    der_act_mat = der_of_activation(z, activation)
    der_x_next = der_x_next.reshape((1, der_x_next.size))
    der_x = np.matmul(np.matmul(der_x_next,der_act_mat),weight)

    return der_x


def rgap_one_layer(in_shape, weight_circ, z_shape, Dz, Dw, inv_act, stride, padding):
    def pseudo_inverse(u,v):
        import numpy as np
        """
        return the pseudo-inverse of ux = v using svd
        """
        result = np.linalg.lstsq(u, v, rcond=None)[0]
        return result

    print("Solving the linear system for pseudo-inverse!\n")
    u, v = get_U_V_conv(in_shape, weight_circ, z_shape, Dz, Dw, inv_act, stride, padding)

    x_pseinv = pseudo_inverse(u.toarray(),v.toarray())
    print("Got pseudo-inverse!\n")

    return x_pseinv, u.toarray(), v.toarray()


def rgap_one_layer_v2(in_shape, weight, der_z, g, inv_act, stride, padding):
    def pseudo_inverse(u,v):
        import numpy as np
        """
        return the pseudo-inverse of ux = v using svd. Padding implementation uses that from R-GAP.
        """
        result = np.linalg.lstsq(u, v, rcond=None)[0]
        return result

    print("Solving the linear system for pseudo-inverse!\n")
    u,v, rank_def = get_U_V_conv_v2(in_shape,  weight, der_z, g, inv_act, stride, padding)
    print("--- The condition number of u is {} ---\n".format(np.linalg.cond(u.toarray())))
    x_pseinv = pseudo_inverse(u.toarray(), v.toarray())
    print("Got pseudo-inverse!\n")

    return x_pseinv[peeling(in_shape=in_shape, padding=padding)], rank_def


def pseudo_inverse(u,v):
    import numpy as np
    """
    return the pseudo-inverse of ux = v using svd, plus u and v as numpy arrays, based on v2
    """
    result = np.linalg.lstsq(u, v, rcond=None)[0]
    return result


def hybrid_solve(module_list, act_list, model_info, n_iter_dlg, n_iter_dlg_1st, device):
    """
    Parameters
    n_iter_dlg: number of iterations for the DLG step:
    Return: estimated input to the network
    """
    print(" ----- Running the hybrid method ----- \n")
    import time, torch
    start_time = time.time()

    der_x_list = []
    weight_list = make_w_circulant(module_list,model_info) # natural order
    num = len(module_list) - 1

    reconstruction = [model_info['target_label']]  # initial input is set to be the target label

    # reconstructing input to the fc layer, which is assumed to be the last in the network
    weight_circ = weight_list[num]
    grad_weight = model_info[str(num)]['grad_weight']
    grad_bias = model_info[str(num)]['grad_bias']
    x, der_x = fully_connected_invert(grad_weight=grad_weight, grad_bias=grad_bias, weight=weight_circ)
    reconstruction.append(x)
    der_x_list.append(der_x)

    # reconstructing from the rest of the network
    for i in range(len(module_list)-2, -1, -1):
        print(" --- Reconstructing layer {} --- \n".format(i))
        z = inv_of_activation(input=reconstruction[-1], act_type=model_info[str(i)]['act_type'])

        # compute der x
        der_x = conv_layer_der_x(der_x_next=der_x_list[-1], weight=weight_list[i], z=z,
                                 activation=model_info[str(i)]['act_type'])
        # compute grad w.r.t z
        der_act_z = der_of_activation(input=z, act_type=model_info[str(i)]['act_type'])
        der_z = np.matmul(der_x_list[-1],der_act_z)

        x_pseinv, u, v = rgap_one_layer(in_shape=model_info[str(i)]['input_shape'], weight_circ=weight_list[i],
                                        z_shape=model_info[str(i)]['z_shape'][1:], Dz=der_z, Dw=model_info[str(i)]['grad_weight'],
                                        inv_act=z, stride=model_info[str(i)]['stride'], padding=model_info[str(i)]['padding'])

        # correct the pseudo inverse using DLG
        # here we need x_true to compute the gradients from the ground truth input
        if i == 0:
            torch.manual_seed(10)
            np.random.seed(10)
            x_corr = dynamic_dlg_soft_constr(u=torch.tensor(u, device=device, requires_grad=False),
                                             v=torch.tensor(v, device=device, requires_grad=False),
                                             x_true=model_info[str(i)]['input_to_layer'],
                                             x_init=torch.tensor(x_pseinv, dtype=torch.float32).to(device),
                                             input_shape=model_info[str(i)]['input_shape'],
                                             y=model_info['target_label'], n_iter=n_iter_dlg_1st,
                                             module_list=module_list, act_list=act_list, index_layer=i,
                                             weight_cossim=1.0, weight_tv=1.0, weight_sqnorm=0.05, #0.05
                                             weight_norm_x=0)

        elif i == 1:
            torch.manual_seed(10)
            np.random.seed(10)
            x_corr = dynamic_dlg_soft_constr_with_rgap(u=torch.tensor(u, device=device, requires_grad=False),
                                                       v=torch.tensor(v, device=device, requires_grad=False),
                                                       x_true=model_info[str(i)]['input_to_layer'],
                                                       x_init=torch.tensor(x_pseinv, dtype=torch.float32).to(device),
                                                       input_shape=model_info[str(i)]['input_shape'],
                                                       y=model_info['target_label'], n_iter=4*n_iter_dlg, # 4*n_iter_dlg
                                                       module_list=module_list, act_list=act_list, index_layer=i,
                                                       weight_cossim=1.0, weight_tv=1.0, weight_sqnorm=0.1,
                                                       weight_norm_x=0)
        elif i >= 2:
            torch.manual_seed(10)
            np.random.seed(10)
            x_corr = dynamic_dlg_soft_constr_with_rgap(u=torch.tensor(u, device=device, requires_grad=False),
                                                       v=torch.tensor(v, device=device, requires_grad=False),
                                                       x_true=model_info[str(i)]['input_to_layer'],
                                                       x_init=torch.tensor(x_pseinv, dtype=torch.float32).to(device),
                                                       input_shape=model_info[str(i)]['input_shape'],
                                                       y=model_info['target_label'], n_iter=n_iter_dlg // 2,
                                                       module_list=module_list, act_list=act_list, index_layer=i,
                                                       weight_cossim=10.0, weight_tv=0.1, weight_sqnorm=1.0,
                                                       weight_norm_x=0)

        reconstruction.append(x_corr)
        der_x_list.append(der_x)
    print("Hybrid method finishes using {} seconds\n".format(time.time() - start_time))
    return reconstruction


def rgap(module_list, model_info):
    print(" ----- Running R-GAP ----- \n")
    import time
    start_time = time.time()
    der_x_list = []
    layer_rank_def = []
    net_rank = 0
    weight_list = make_w_circulant(module_list,model_info) # natural order
    num = len(module_list) - 1

    reconstruction = [model_info['target_label']]  # inital input is set to be the target label

    # reconstructing input to the fc layer, which is assumed to be the last in the network
    weight_circ = weight_list[num]
    grad_weight = model_info[str(num)]['grad_weight']
    grad_bias = model_info[str(num)]['grad_bias']
    x, der_x = fully_connected_invert(grad_weight=grad_weight, grad_bias=grad_bias, weight=weight_circ)
    reconstruction.append(x)
    der_x_list.append(der_x)

    # reconstructing from the rest of the network
    for i in range(len(module_list)-2, -1, -1):
        print(" --- Reconstructing layer {} --- \n".format(i))
        z = inv_of_activation(input=reconstruction[-1], act_type=model_info[str(i)]['act_type'])
        # compute der x
        der_x = conv_layer_der_x(der_x_next=der_x_list[-1], weight=weight_list[i], z=z,
                                 activation=model_info[str(i)]['act_type'])
        # compute grad w.r.t z
        der_act_z = der_of_activation(input=z, act_type=model_info[str(i)]['act_type'])
        der_z = np.matmul(der_x_list[-1],der_act_z)

        x_pseinv, rank_def = rgap_one_layer_v2(in_shape=model_info[str(i)]['input_shape'],
                                               weight=model_info[str(i)]['weight'],
                                               der_z=der_z, g=model_info[str(i)]['grad_weight'],
                                               inv_act=z, stride=model_info[str(i)]['stride'],
                                               padding=model_info[str(i)]['padding'])

        print(" --- Rank deficiency for layer {} is {} --- \n".format(i, rank_def))
        reconstruction.append(x_pseinv)
        der_x_list.append(der_x)
        layer_rank_def.append(rank_def)
        net_rank += round(rank_def*1.0*(len(module_list) - i - 1) / (len(module_list)-1))
    print("R-GAP method finishes using {} seconds\n".format(time.time() - start_time))
    print("The COPA index for the model is {}\n".format(net_rank))
    return reconstruction


def dynamic_forward(x_input, module_list, activation_list, index_layer):
    import torch
    modules = module_list[index_layer:]
    acts = activation_list[index_layer:]
    z = x_input
    para_list = []
    for (layer, act) in list(zip(modules, acts)):
        if isinstance(layer, torch.nn.Linear):
            z = torch.flatten(z,1)
        layer_out = layer(z)
        act_out = act(layer_out)
        z = act_out
        if not isinstance(layer.bias, type(None)):
            para_list.append(layer.bias)
        para_list.append(layer.weight)

    return z, modules, acts, para_list


def dlg_orig(x_orig, x_init, y, n_iter, model):
    """
    re-implementing DLG based on https://github.com/mit-han-lab/dlg
    """
    import torch, time
    print(" ----- Running original DLG ----- \n")
    start_time = time.time()
    x = x_init.detach().clone()
    x.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()

    pred = model(x_orig)[0]
    loss = criterion(pred, y)
    dy_dx = torch.autograd.grad(loss, model.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    optimizer = torch.optim.LBFGS([x])

    for iters in range(n_iter):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(x)[0]
            dummy_loss = criterion(dummy_pred, y)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)

        if iters % 10 == 0:
            current_loss = closure().item()
            print("iters {}, loss (grad diff) : {:.4f}\n".format(iters, current_loss))
    print("Original DLG finished in {} seconds".format(time.time()-start_time))
    return x


def CosDistTV(x_orig, x_init, y, n_iter, model, weight_TV):
    """
    re-implementing "How easy is it to break privacy in federated learning?" Geiping et al, NeurIPS 20.
    """
    import torch, time
    print(" ----- Running DLG with cosine distance ----- \n")
    start_time = time.time()
    x = x_init.detach().clone()
    x.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()

    pred = model(x_orig)[0]
    loss = criterion(pred, y)
    dy_dx = torch.autograd.grad(loss, model.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    optimizer = torch.optim.Adam([x])

    for iters in range(n_iter):
        def closure():
            optimizer.zero_grad()

            dummy_pred = model(x)[0]
            dummy_loss = criterion(dummy_pred, y)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            cosdist = 0
            for grad_1, grad_2 in zip(dummy_dy_dx, original_dy_dx):
                cosdist += (1 - torch.nn.functional.cosine_similarity(grad_1.flatten(), grad_2.flatten(), 0, 1e-10)) +\
                            weight_TV*total_variation(x)

            cosdist.backward()
            return cosdist

        optimizer.step(closure)

        if iters % 50 == 0:
            current_loss = closure().item()
            print("iters {}, loss (grad diff) : {:.4f}\n".format(iters, current_loss))
    print("DLG with cosine distance finished in {} seconds".format(time.time()-start_time))
    return x


def dynamic_dlg_soft_constr(u, v, x_true, x_init, input_shape, y, n_iter, module_list, act_list, index_layer,
                            weight_cossim, weight_tv, weight_sqnorm, weight_norm_x):
    """
    implenting the method from "Deep Leakage from Gradients", https://github.com/mit-han-lab/dlg
    Using constrained optimisation to enforce the condition defined by the linear system.
    x_init: initialisation of the input
    y: true label
    """
    import torch, copy
    print(" --- Optimising Cossim with TV --- \n")

    x = x_init.detach().clone()
    x.requires_grad = True
    optimizer = torch.optim.Adam([x])
    criterion = torch.nn.CrossEntropyLoss()
    true_pred, _, _, true_para_list = dynamic_forward(x_input=x_true, module_list=module_list, activation_list=act_list,
                                                      index_layer=index_layer)
    true_loss = criterion(true_pred, y)
    true_grads_ = torch.autograd.grad(true_loss, true_para_list)
    true_grads = [_.detach().clone() for _ in true_grads_]

    for iters in range(n_iter):
        def closure():
            optimizer.zero_grad()
            dummy_pred, _, _, dummy_para_list = dynamic_forward(x_input=x.reshape(input_shape), module_list=module_list,
                                                                activation_list=act_list, index_layer=index_layer)

            dummy_loss = criterion(dummy_pred, y)
            dummy_grads = torch.autograd.grad(dummy_loss, dummy_para_list, create_graph=True)
            combined_obj = 0
            cossim = 0
            for grad_1, grad_2 in zip(dummy_grads, true_grads):
                cossim += (1 - torch.nn.functional.cosine_similarity(grad_1.flatten(),
                                                                     grad_2.flatten(),
                                                                     0, 1e-10))
            combined_obj += weight_cossim*cossim + weight_tv*total_variation(x.reshape(input_shape)) + \
                            weight_sqnorm*square_norm(u=copy.deepcopy(u), v=copy.deepcopy(v), x=x) + \
                            weight_norm_x*x.transpose(0,1).matmul(x)
            combined_obj.backward()
            return combined_obj

        optimizer.step(closure)
        if iters % 50 == 0:
            curr_loss = closure().item()
            print("iters {}, loss, tv : {:.4f}, {}\n".format(iters, curr_loss, total_variation(x.reshape(input_shape))))
    print("Dynamic DLG finished!\n")
    return x.detach().clone().cpu().numpy()


def dynamic_dlg_soft_constr_with_rgap(u, v, x_true, x_init, input_shape, y, n_iter, module_list, act_list, index_layer,
                                      weight_cossim, weight_tv, weight_sqnorm, weight_norm_x):
    """
    implenting the method from "Deep Leakage from Gradients", https://github.com/mit-han-lab/dlg
    Write the squared norm given by the linear system as a summand in the objective function.
    x_init: initialisation of the input
    y: true label
    """
    import torch, copy
    print(" --- Running our dynamic DLG with R-GAP as soft constraint --- \n")

    x = x_init.detach().clone()
    x.requires_grad = True
    optimizer = torch.optim.Adam([x])
    criterion = torch.nn.CrossEntropyLoss()
    true_pred, _, _, true_para_list = dynamic_forward(x_input=x_true, module_list=module_list, activation_list=act_list,
                                                      index_layer=index_layer)
    true_loss = criterion(true_pred, y)
    true_grads_ = torch.autograd.grad(true_loss, true_para_list)
    true_grads = [_.detach().clone() for _ in true_grads_]

    for iters in range(n_iter):
        def closure():
            optimizer.zero_grad()
            dummy_pred, _, _, dummy_para_list = dynamic_forward(x_input=x.reshape(input_shape), module_list=module_list,
                                                                activation_list=act_list, index_layer=index_layer)

            dummy_loss = criterion(dummy_pred, y)
            dummy_grads = torch.autograd.grad(dummy_loss, dummy_para_list, create_graph=True)
            combined_obj = 0
            cossim = 0
            for grad_1, grad_2 in zip(dummy_grads, true_grads):
                cossim += (1 - torch.nn.functional.cosine_similarity(grad_1.flatten(),
                                                                     grad_2.flatten(),
                                                                     0, 1e-10))
            combined_obj += weight_cossim*cossim + weight_tv*total_variation(x.reshape(input_shape)) + \
                            weight_sqnorm*square_norm(u=copy.deepcopy(u), v=copy.deepcopy(v), x=x) + \
                            weight_norm_x*x.transpose(0,1).matmul(x)
            combined_obj.backward()
            return combined_obj

        optimizer.step(closure)
        if iters % 50 == 0:
            curr_loss = closure().item()
            print("iters {}, loss, tv : {:.4f}, {}\n".format(iters, curr_loss, total_variation(x.reshape(input_shape))))
    print("Dynamic DLG finished!\n")
    return x.detach().clone().cpu().numpy()


def compute_grad_diff(x_init, x_true, y, module_list, index_layer, act_list, device, input_shape):
    """
    compute the L2 norm of the gradient difference
    """
    import torch
    x = (torch.tensor(x_init, dtype=torch.float32)).clone().detach().to(device)
    x.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()

    # compute gradients from the target
    true_pred, _, _, true_para_list = dynamic_forward(x_input=x_true, module_list=module_list, activation_list=act_list,
                                                      index_layer=index_layer)
    true_loss = criterion(true_pred, y)
    true_grads_ = torch.autograd.grad(true_loss, true_para_list)
    true_grads = [_.detach().clone() for _ in true_grads_]

    # compute gradients from the variable x_init
    dummy_pred, _, _, dummy_para_list = dynamic_forward(x_input=x.reshape(input_shape), module_list=module_list,
                                                        activation_list=act_list, index_layer=index_layer)
    dummy_loss = criterion(dummy_pred, y)
    dummy_grads_ = torch.autograd.grad(dummy_loss, dummy_para_list)
    dummy_grads = [grad.detach().clone() for grad in dummy_grads_]
    grad_diff = 0
    # L2 norm of the grad diff
    for grad_1, grad_2 in zip(dummy_grads, true_grads):
        grad_diff += ((grad_1 - grad_2) ** 2).sum()
    return grad_diff.detach().clone().cpu().numpy()


def square_norm(u, v, x):
    """
    u,v,x: torch tensor, can be used as part of the objective function in a gradient based optimisation such
    as ADAM
    """
    x_casted = x.type(u.dtype)
    temp = u.matmul(x_casted) - v
    result = temp.transpose(0,1).matmul(temp)
    return result[0][0]


def comparisons():
    import torch
    output_path = '/home/cangxiong/projects/dp/experiments/Grad_invert_attacks/output/' # path for saving the plots
    data_path = '/mnt/storage/datasets/cifar10/' # path to the CIFAR10 dataset; will be downloaded if it does not exist
    # 'Y/N' indicates whether to include the corresponding model in the comparison
    config = {'DLG': 'Y', # whether to include DLG method
              'DLG_iter': 1, # number of iterations for DLG, default 300
              'Geiping': 'Y', # whether to include Geiping et al
              'Geiping_iter': 1, # number of iterations for Geiping et al, default 4800
              'RGAP': 'Y', # whether to include RGAP
              'HYB': 'Y',  # whether to include our hybrid method
              'DYN_dlg_iter': 1, # iteration for all but the first layer, taken to be 2000 in the paper
              'DYN_dlg_iter_1st': 1, # iteration for reconstructing the first layer, taken to be 10000 in the paper
              'target_model': 'cnn3_c1', # target model architecture, see targets.py for available models
              'datapoint_index': 3, # choose which image to use for the comparison
              'note': '' # additional note for the output file (optional)
              }

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    trainloader = test_on_CIFAR10(data_path=data_path, index=config['datapoint_index'])
    data_list = []
    for i, data in enumerate(trainloader):
        data_list.append((data[0].to(device),data[1].to(device)))
    image = data_list[0][0][:,:,:32,:32]
    label = data_list[0][1]
    torch.manual_seed(10)
    np.random.seed(10)
    model = cnn4_c2(device=device)

    module_list = model.module_list
    act_list = model.activation_list
    model_info = create_model_info(model=model,input=image,label=label,loss_fn=torch.nn.CrossEntropyLoss())

    if config['HYB'] == 'Y':
        hybrid_recon = hybrid_solve(module_list=module_list, act_list=act_list, model_info=model_info,
                                    n_iter_dlg=config['DYN_dlg_iter'], n_iter_dlg_1st=config['DYN_dlg_iter_1st'],
                                    device=device)[-1]
        hybrid_recon = hybrid_recon.reshape(image.shape)
    elif config['HYB'] == 'N':
        hybrid_recon = np.ones(shape=(1,3,32,32))

    if config['DLG'] == 'Y':
        torch.manual_seed(10)
        np.random.seed(10)
        dlg_init_input = torch.randn(image.size()).to(device)
        torch.manual_seed(10)
        np.random.seed(10)
        dlg_recon = dlg_orig(x_orig=image, x_init=dlg_init_input, y=label, n_iter=config['DLG_iter'], model=model)
        dlg_recon = dlg_recon.detach().cpu().numpy()
    elif config['DLG'] == 'N':
        dlg_recon = np.ones(shape=(1,3,32,32))

    if config['Geiping'] == 'Y':
        torch.manual_seed(10)
        np.random.seed(10)
        geiping_init = torch.randn(image.size()).to(device)
        geiping_recon = CosDistTV(x_orig=image, x_init=geiping_init, y=label, n_iter=config['Geiping_iter'], model=model,
                                  weight_TV=0.01)
        geiping_recon = geiping_recon.detach().cpu().numpy()
    elif config['Geiping'] == 'N':
        geiping_recon = np.ones(shape=(1,3,32,32))

    if config['RGAP'] == 'Y':
        rgap_recon = rgap(module_list, model_info)[-1]
        rgap_recon = rgap_recon.reshape(image.shape)
    elif config['RGAP'] == 'N':
        rgap_recon = np.ones(shape=(1,3,32,32))

    print(" ----- Test name is : {} ----- ".format(config['target_model'] + '_' + str(config['datapoint_index']) + config['note']))
    plot_compare(orig_img=image, rgap_recon=rgap_recon, dlg_recon=dlg_recon, hybrid_recon=hybrid_recon,
                 geiping_recon=geiping_recon, root_directory=output_path,
                 test_name=config['target_model'] + '_' + str(config['datapoint_index']) + config['note'])


def main():
    comparisons()

if __name__ == '__main__':
    main()


