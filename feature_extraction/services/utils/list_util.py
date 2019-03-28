def unwrap_from_np_array(lis_of_np_array):
    list_of_lists = [x.tolist() for x in lis_of_np_array]
    return [item for sublist in list_of_lists for item in sublist]


def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
