

# Some of the code here was written by Boyuan (John) Yao

import hippylib as hp
import hippyflow as hf
import numpy as np

def operator_to_array_with_dummy_vectors(A, input_vector, output_vector):
    """
    Pre-multiply operator against the standard basis to get its components,
    Uses input and output vectors provided to help with initialization.

    :param A: The operator to convert with method :code:`mult(x,y)`
    :param input_vector: A vector on the input space
    :type input_vector: dl.Vector

    :param output_vector: A vector on the output space
    :type output_vector: dl.Vector

    :returns: The :code:`np.ndarray` representation of operator :code:`A`

    .. note:: Can work with serial mesh but parallel sampling
    """
    dim_input = input_vector.get_local().shape[0]
    basis_all = hp.MultiVector(input_vector, dim_input)
    A_basis_all = hp.MultiVector(output_vector, dim_input)
    identity = np.eye(dim_input)
    for i in range(dim_input):
        basis_all[i].set_local(identity[:,i])
        # print("Multiplying basis %d of %d" %(i+1, dim_input))

    hp.MatMvMult(A, basis_all, A_basis_all)
    return hf.mv_to_dense(A_basis_all)