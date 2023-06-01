class ListUtils(object):
    pass

    @staticmethod
    def _nest_list(flat_list, nested_size):
        return [flat_list[i:i + nested_size] for i in range(0, len(flat_list), nested_size)]