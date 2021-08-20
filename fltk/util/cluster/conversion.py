import re

class Convert:
    """
    @deprecated: This file is up for removal or change.
    Class for conversion of K8s memory and cpu descriptions. Based on teh implementatino by amelbakry
    https://github.com/amelbakry/kube-node-utilization/blob/0afc529eab0199b7746ea0a50aa76ed23cb0ba3f/nodeutilization.py#L18-L46
    """

    _mem_dict = {
        re.compile(r"[0-9]{1,9}Mi?"): lambda x: int(re.sub("[^0-9]", "", x)),
        re.compile(r"[0-9]{1,9}Ki?"): lambda x: int(re.sub("[^0-9]", "", x)) // 1024,
        re.compile(r"[0-9]{1,9}Gi?"): lambda x: int(re.sub("[^0-9]", "", x)) * 1024
    }

    _cpu_dict = {
        re.compile(r"[0-9]{1,4}$"): lambda x: int(re.sub("[^0-9]", "", x)) * 1e3,  # cores
        re.compile(r"[0-9]{1,9}m"): lambda x: int(re.sub("[^0-9]", "", x)),  # milli cores
        re.compile(r"[0-9]{1,15}u"): lambda x: int(re.sub("[^0-9]", "", x)) // 1e3,  # micro cores
        re.compile(r"[0-9]{1,15}n"): lambda x: int(re.sub("[^0-9]", "", x)) // 1e6  # nano cores
    }

    @staticmethod
    def __convert(dictionary, value):
        """
        Mapping function with corresponding dictionary
        @param dictionary:
        @type dictionary:
        @param value:
        @type value:
        @return:
        @rtype:
        """
        for re_expression, mapper in dictionary.items():
            if re_expression.match(value):
                return mapper(value)

    @staticmethod
    def cpu(value):
        """
        Return CPU description in terms of milli cores.
        """
        return Convert.__convert(Convert._cpu_dict, value)

    @staticmethod
    def memory(value: str):
        """
        Convert str representation of memory (e.g. allocatable memory) to integer representation
        in Mega Bytes (MB).
        @param value: str representation of memory.
        @type value:
        @return:
        @rtype:
        """
        return Convert.__convert(Convert._mem_dict, value)


