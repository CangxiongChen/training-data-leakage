# compute circulant representations of weight and gradient matrices
import torch
import sys
import numpy as np


def circulant_rep_single_channel(kernel, input_h, padding, stride, dilation, bias='none'):
    """
    Assuming square kernel.
    X: m by m
    W: k by k (kernel matrix from one channel to another
    B: 1,
    bias: place holder
    """
    import numpy as np
    if dilation == 1:
        kernel_trans = kernel
    elif dilation > 1:
        # initialise kernel_trans
        k = kernel.shape[0]
        k_trans = k + (k-1)*(dilation-1)
        kernel_trans = np.zeros(shape=(k_trans,k_trans))
        for row_ind in range(k):
            for col_ind in range(k):
                kernel_trans[dilation*row_ind, dilation*col_ind] = kernel[row_ind, col_ind]
    # print("Kernel trans is {}, it has shape {}".format(kernel_trans, kernel_trans.shape))
    # first perform the computation without bias, then augment the results adding bias
    m = input_h
    k = kernel_trans.shape[0]
    u = (m + 2*padding - k) // stride + 1  # output size after convolution
    m_aug = m + 2*padding
    v_aug = m_aug**2  # size of the flattened augmented input
    W_circ_aug = np.zeros((u**2,v_aug))

    # create a flattened version of W with zeros inserted at the proper location
    zeros_lst = np.zeros((m_aug - k ,))
    w_mod = np.insert(kernel_trans,[k],zeros_lst,axis=1)

    # cut off the zeros inserted in the last row of the kernel
    if m_aug != k:
        w_mod = w_mod.flatten()[: k - m_aug]
    elif m_aug == k:  # edge case
        w_mod = w_mod.flatten()
    k_mod = w_mod.size
    ind_1 = 0

    # while the replacement W_mod does not go over the bound
    for i in range(u**2):
        np.put(W_circ_aug[i,:],range(ind_1,ind_1+k_mod),w_mod)
        # find the index for the next replacement operation in the next row
        ind_2 = ind_1 + stride + k - 1
        if (ind_2 // m_aug) == (ind_1 // m_aug):
            ind_1 += stride
        elif (ind_2 // m_aug) > (ind_1 // m_aug):
            residue = (m_aug - k) % stride
            ind_1 += (k + (stride-1)*m_aug + residue)
    # slicing to get back the circulant form of W
    slice_ind = []
    start_ind = padding*(m_aug+1)
    end_ind = v_aug - padding*(m_aug+1)
    for ind in range(start_ind, end_ind):
        if ((ind % m_aug) <= padding + m - 1) and (ind % m_aug >= padding):
            slice_ind.append(ind)
    W_circ = W_circ_aug[:,slice_ind]

    # augment W_circ to include bias
    if bias != 'none':
        W_circ = np.hstack([W_circ, np.full(shape=(W_circ.shape[0],1),fill_value=bias)])

    return W_circ


def circulant_rep(weight, input_h, padding, stride, dilation, bias='none'):
    """
    weight: 4 dimension (output channel, input channel, row, col), to be consistent with pytorch
    input_size: per channel size of the input
    """
    from scipy import sparse

    input_channel = weight.shape[1]
    output_channel = weight.shape[0]
    k_orig = weight.shape[2]
    k = k_orig + (k_orig - 1)*(dilation - 1) # width of the weight per channel
    u = (input_h + 2*padding - (k-1) - 1) // stride + 1  # output width per channel after convolution
    output_size = output_channel*(u**2)  # size of the flattened output
    w_circ = sparse.lil_matrix((output_size, (input_h**2)*input_channel))
    for l in range(input_channel):
        for k in range(output_channel):
            kernel = weight[k,l,:,:]
            w_circ_per_channel = circulant_rep_single_channel(kernel, input_h, padding, stride, dilation, bias=bias)
            num_rows = w_circ_per_channel.shape[0]
            num_cols = w_circ_per_channel.shape[1]
            w_circ[k*num_rows: (k+1)*num_rows, l*num_cols:(l+1)*num_cols] = w_circ_per_channel
    return w_circ


def circ_rep_grad(weight_shape, der_z, x_shape, stride, padding):
    """
    Using existing circ rep for w.
    x_shape:  (number of datapoints, channels, row, column), same as PyTorch's convention
    der_z must be of z_shape, i.e. (k, m, m) where k is the number of output channels
    weight_shape: (out,in,height,width) --> numpy array
    """
    from scipy import sparse
    # setting the input and output dimensions of the representation
    # input must be the same as that of x_shape (flattened)
    in_dim = x_shape[1]*(x_shape[2]**2)

    # output dimension must be the same as gradients of weights
    weight_size = np.prod(weight_shape)
    out_dim = weight_size

    # compute the effective height of the input
    residue = (x_shape[2] + 2*padding - weight_shape[2]) % stride
    h_eff = x_shape[2] - residue
    h = x_shape[2]
    identity_mat = sparse.identity(n=x_shape[2]**2,format='csr')

    # initialise an empty sparse matrix
    dz_circ = sparse.lil_matrix((out_dim,in_dim))
    for row_ind in range(weight_shape[0]): # over the output dim
        # first compute the single channel representation, which can be treated as a convolution (dilation will
        # correspond to stride in the convolution)

        der_circ_one_chan = circulant_rep_single_channel(kernel=der_z[row_ind,:,:], input_h=h_eff,
                                                            padding=padding, stride=1, dilation=stride, bias='none')

        # define the projection matrix which projects x to a submatrix with size h_eff & h_eff
        rows_indices = [i for i in range(x_shape[2]**2)]
        indices_to_delete = []
        for i in range(h-residue, h, 1):
            for j in range(h):
                indices_to_delete.append(j*h + i)
                indices_to_delete.append(i*h + j)
        rows_indices_remain = np.delete(rows_indices, indices_to_delete)
        P = identity_mat[rows_indices_remain]
        # multiply with the projection matrix
        der_circ_one_chan_composed = sparse.csr_matrix(der_circ_one_chan).dot(P)
        mat_copies = [der_circ_one_chan_composed for i in range(weight_shape[1])]
        block_mat = sparse.block_diag(mat_copies, format='lil')
        # replace the corresponding block in dz_circ
        dz_circ[row_ind*block_mat.shape[0]:(row_ind+1)*block_mat.shape[0], :] = block_mat

    return dz_circ














