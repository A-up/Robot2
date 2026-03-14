# Поворот роборуки с помощью блоков установки скважности для ШИМ контроллера
import sys
from ctypes import *
import time
from ctypes.util import find_library
import platform
import threading

class Servo:
    def __init__(self, lib, sid, pwm):
        self.sid = sid
        self.pwm = pwm
        self.lib = lib
        self.errTextC = create_string_buffer(1000)
    def SetSyncServoRotation(self, new_rotation):
        errCode = self.lib.RI_SDK_sigmod_PWM_SetPortDutyCycle(self.pwm, self.sid, 0, int(390+new_rotation*2.8888888), self.errTextC)
        if errCode != 0:
            print(errCode, self.errTextC.raw.decode())
            sys.exit(2)
        time.sleep(0.25)


    def SetAsyncServoRotation(self, new_rotation):
        th = threading.Thread(target=self.SetSyncServoRotation, args=[float(new_rotation)])
        th.start()

# Подключаем внешнюю библиотеку для работы с SDK
class Manipulator:

    def __init__(self):
        libName = ""
        plat = platform.system()
        if plat == "Windows":
            libName = "librisdk.dll"
        if plat == "Linux":
            libName = "librisdk.so"

        pathLib = find_library(libName)
        self.lib = cdll.LoadLibrary(pathLib)

        # Указываем типы аргументов для функций библиотеки RI_SDK
        self.lib.RI_SDK_InitSDK.argtypes = [c_int, c_char_p]
        self.lib.RI_SDK_CreateModelComponent.argtypes = [c_char_p, c_char_p, c_char_p, POINTER(c_int), c_char_p]
        self.lib.RI_SDK_LinkPWMToController.argtypes = [c_int, c_int, c_uint8, c_char_p]
        self.lib.RI_SDK_DestroySDK.argtypes = [c_bool, c_char_p]
        self.lib.RI_SDK_sigmod_PWM_SetPortDutyCycle.argtypes = [c_int, c_int, c_int, c_int, c_char_p]


        self.errTextC = create_string_buffer(1000)  # Текст ошибки. C type: char*
        self.i2c = c_int()
        self.pwm = c_int()

        # Инициализация библиотеки RI SDK с уровнем логирования 3
        errCode = self.lib.RI_SDK_InitSDK(3, self.errTextC)
        if errCode != 0:
            print(errCode, self.errTextC.raw.decode())
            sys.exit(2)

        # Создание компонента i2c адаптера модели ch341
        errCode = self.lib.RI_SDK_CreateModelComponent("connector".encode(), "i2c_adapter".encode(), "ch341".encode(), self.i2c,
                                                  self.errTextC)
        if errCode != 0:
            print(errCode, self.errTextC.raw.decode())
            sys.exit(2)

        # Создание компонента ШИМ модели pca9685
        errCode = self.lib.RI_SDK_CreateModelComponent("connector".encode(), "pwm".encode(), "pca9685".encode(), self.pwm, self.errTextC)
        if errCode != 0:
            print(errCode, self.errTextC.raw.decode())
            sys.exit(2)

        # Связывание i2c с ШИМ
        errCode = self.lib.RI_SDK_LinkPWMToController(self.pwm, self.i2c, 0x40, self.errTextC)
        if errCode != 0:
            print(errCode, self.errTextC.raw.decode())
            sys.exit(2)
        self.Base = Servo(self.lib, 0, self.pwm)
        self.Hand = Servo(self.lib, 1, self.pwm)
        self.Up = Servo(self.lib, 3, self.pwm)
        self.Side = Servo(self.lib, 2, self.pwm)
        self.RHand = Servo(self.lib, 4, self.pwm)


    def __del__(self):
        # Удаление библиотеки со всеми компонентами
        errCode = self.lib.RI_SDK_DestroySDK(True, self.errTextC)
        if errCode != 0:
            print(errCode, self.errTextC.raw.decode())
            sys.exit(2)

