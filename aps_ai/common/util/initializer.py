# #########################################################################
# Copyright (c) 2020, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2020. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################
from aps_ai.common.util import Singleton, synchronized_method

import os
from configparser import ConfigParser

class IniMode:
    LOCAL_FILE = 0
    REMOTE_FILE = 1
    DATABASE = 2
    NONE = 99

class AlreadyInitializedError(ValueError):
    def __init__(self, message=None):
        super(AlreadyInitializedError, self).__init__(message)

class GenericRegistry(object):
    _NO_APPLICATION = "<NO APPLICATION>"

    def __init__(self, registry_name):
        self.__registry_name = registry_name
        self.__registry = {self._NO_APPLICATION: None}

    def register_instance(self, instance, application_name=None, replace=False):
        if instance is None: raise ValueError(self.__registry_name + " Instance is None")

        application_name = self.__get_application_name(application_name)

        if application_name in self.__registry.keys():
            if self.__registry[application_name] is None or replace==True:
                self.__registry[application_name] = instance
            else:
                raise AlreadyInitializedError(self.__registry_name + " Instance already initialized")
        else:
            self.__registry[application_name] = instance

    def reset(self, application_name=None):
        application_name = self.__get_application_name(application_name)

        if application_name in self.__registry.keys():
            self.__registry[self.__get_application_name(application_name)] = None
        else:
            raise ValueError(self.__registry_name + " Instance not existing")

    def get_instance(self, application_name=None):
        application_name = self.__get_application_name(application_name)

        if application_name in self.__registry.keys():
            return self.__registry[self.__get_application_name(application_name)]
        else:
            raise ValueError(self.__registry_name + " Instance not existing")

    def __get_application_name(self, application_name):
        return self._NO_APPLICATION if application_name is None else application_name


class IniFacade:
    def set_value_at_ini(self, section, key, value): raise NotImplementedError()
    def set_list_at_ini(self, section, key, values_list=[]): raise NotImplementedError()
    def get_string_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_int_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_float_from_ini(self, section, key, default=None): raise NotImplementedError()
    def get_boolean_from_ini(self, section, key, default=False): raise NotImplementedError()
    def get_list_from_ini(self, section, key, default=None): raise NotImplementedError()
    def dump(self): raise NotImplementedError()
    def push(self): raise NotImplementedError()

class __NullIni(IniFacade):
    def set_value_at_ini(self, section, key, value): pass
    def set_list_at_ini(self, section, key, values_list=[]): pass
    def get_string_from_ini(self, section, key, default=None): pass
    def get_int_from_ini(self, section, key, default=None): pass
    def get_float_from_ini(self, section, key, default=None): pass
    def get_boolean_from_ini(self, section, key, default=False): pass
    def get_list_from_ini(self, section, key, default=None): pass
    def dump(self): pass
    def push(self): pass

class __LocalIniFile(IniFacade):
    def __init__(self, **kwargs):
        self.__ini_file_name = kwargs["ini_file_name"]

        if not os.path.isfile(self.__ini_file_name):
            with open(self.__ini_file_name, "w") as ini_file: ini_file.write('[Files]\n\n\n[Parameters]\n')
            print("File " + self.__ini_file_name + " doesn't exist: created empty ini file.")

        self.__config_parser = ConfigParser()
        self.__config_parser.read(self.__ini_file_name)

    def get_ini_file_name(self):
        return self.__ini_file_name

    def __get_from_ini(self, section, key, default=None):
        try:
            value = self.__config_parser[section][key]
            value = value.strip()
            return None if value.lower() == "none" else value
        except:
            return str(default) if not default is None else None

    def set_value_at_ini(self, section, key, value):
        try:
            self.__config_parser[section][key] = "None" if value is None else str(value)
        except:
            if not self.__config_parser.has_section(section): self.__config_parser.add_section(section)
            if not self.__config_parser.has_option(section, key): self.__config_parser.set(section, key, "None" if value is None else str(value))

    def set_list_at_ini(self, section, key, values_list=[]):
        if values_list is None: values_string = "None"
        else:
            values_string = ""
            for value in values_list: values_string += str(value) + ", "
            values_string = values_string[:-2]

        self.set_value_at_ini(section, key, values_string)

    def get_string_from_ini(self, section, key, default="None"):
        value = self.__get_from_ini(section, key, default)
        return str(default) if value is None else value.strip()

    def get_int_from_ini(self, section, key, default=0):
        value = self.__get_from_ini(section, key, default)
        return int(default) if value is None else int(value.strip())

    def get_float_from_ini(self, section, key, default=0.0):
        value = self.__get_from_ini(section, key, default)
        return float(default) if value is None else float(value.strip())

    def get_boolean_from_ini(self, section, key, default=False):
        value = self.__get_from_ini(section, key, default)
        return default if value is None else (True if value.strip().lower() == "true" else False)

    def get_list_from_ini(self, section, key, default=[]):
        value = self.__get_from_ini(section, key, default=None)
        return default if value is None else list(map(int, self.__get_from_ini(section, key).split(',')))

    def dump(self):
        text = "Dump of file: " + self.__ini_file_name + "\n" + \
               "%============================================================"

        for section in self.__config_parser.sections():
            text += "\n[" + section + "]\n"

            for option in self.__config_parser.options(section):
                text += option + " = " + str(self.__config_parser.get(section, option)) + "\n"

        text += "%============================================================\n"

        return text

    def push(self):
        with open(self.__ini_file_name, "w") as ini_file: self.__config_parser.write(ini_file)

@Singleton
class __IniRegistry(GenericRegistry):
    def __init__(self):
        GenericRegistry.__init__(self, registry_name="Ini")

    @synchronized_method
    def register_ini(self, ini_facade_instance, application_name=None):
        super().register_instance(ini_facade_instance, application_name, False)

    @synchronized_method
    def reset(self, application_name=None):
        super().reset(application_name)

    def get_ini_instance(self, application_name=None):
        return super().get_instance(application_name)


# -----------------------------------------------------
# Factory Methods

def register_ini_instance(ini_mode=IniMode.LOCAL_FILE, reset=False, application_name=None, **kwargs):
    if reset: __IniRegistry.Instance().reset(application_name)
    if ini_mode == IniMode.LOCAL_FILE: __IniRegistry.Instance().register_ini(__LocalIniFile(**kwargs), application_name)
    elif ini_mode == IniMode.NONE:     __IniRegistry.Instance().register_ini(__NullIni(), application_name)

def get_registered_ini_instance(application_name=None):
    return __IniRegistry.Instance().get_ini_instance(application_name)



