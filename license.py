# ! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import requests
import base64
from Crypto.Cipher import AES
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex
import time
import random
import string
import json
import hashlib


class new_AES(object):
    def __init__(self, key, iv):
        self.key = key
        self.iv = iv
        self.mode = AES.MODE_CBC

    # 如果text不足16位的倍数就用空格补足为16位
    def add_to_16(self, text):
        if len(text.encode('utf-8')) % 16:
            add = 16 - (len(text.encode('utf-8')) % 16)
        else:
            add = 0
        text = text + ('\0' * add)
        return text.encode('utf-8')

    # 加密函数
    def encrypt(self, text):
        text = self.add_to_16(text)
        cryptos = AES.new(self.key, self.mode, self.iv)
        cipher_text = cryptos.encrypt(text)
        # 因为AES加密后的字符串不一定是ascii字符集的，输出保存可能存在问题，所以这里转为16进制字符串
        return b2a_hex(cipher_text)


    # 解密后，去掉补足的空格用strip() 去掉
    def decrypt(self, text):
        cryptos = AES.new(self.key, self.mode, self.iv)
        plain_text = cryptos.decrypt(a2b_hex(text))
        return bytes.decode(plain_text).rstrip('\0')

# 获取机器的uuid
def get_machine_id():
    number = ''
    # 上传机器唯一标识，获取最新版本信息，查看是否需要更新
    if sys.platform.startswith("win"):
        import wmi
        # 取硬盘号作为唯一id
        c = wmi.WMI()
        number = c.Win32_PhysicalMedia()[0].SerialNumber.lstrip().rstrip()
    if sys.platform.startswith("dar"):
        # 取mac下的唯一id
        cmd = "/usr/sbin/system_profiler SPHardwareDataType | fgrep 'Serial' | awk '{print $NF}'"
        number = os.popen(cmd).read().strip()
    if sys.platform.startswith('linux') or sys.platform.startswith('free'):
        # 取linux下的唯一id
        cmd = "ls -l /dev/disk/by-uuid/|awk -F \" \" '{print $9}'"
        number = os.popen(cmd).read().strip()
    if not number:
        number = 'error'
    # 向官网发送请求 获取最新版本
    data = {"machine_id": number, "version": 1, "platform": sys.platform}
    return data

# 注册license
def regist_license(license, uuid):
    key = '10100202 etimreh'.encode('utf-8')
    iv = time.strftime("%Y-%m-%d %H-00", time.gmtime()).encode('utf-8')
    tmp_AES = new_AES(key, iv)
    data = {'license':license, 'uuid':uuid}
    base_url = ""
    random_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))
    data['random_str'] = random_str
    # 求字符串MD5
    md5 = hashlib.md5(str(data).encode('utf-8')).hexdigest()
    data['md5'] = md5
    # {'license': '11111', 'uuid': {'machine_id': '0025_3884_01B7_CA8C.', 'version': 1, 'platform': 'win32'},
    # 'random_str': 'Hdk7XQRsE5aP36oY', 'md5': '354f3d1fe7b3a4a0927b3c1320817230'}
    #print(str(data), iv)
    encrypt_data = tmp_AES.encrypt(str(data))
    # return encrypt_data
    res = requests.post(url="http://caokai.deepmd.net/data/license/check", data=encrypt_data)
    #res = requests.post(url="http://127.0.0.1:5000/data/license/check", data=encrypt_data)
    # res = json.loads(res.text)
    #print(res.text)
    return res.text, random_str


def decode(encrypt_data, key):
    iv = '10100202 etimreh'.encode('utf-8')
    cryptor = new_AES(key, iv)
    plain_text = cryptor.decrypt(encrypt_data)
    return plain_text


def test_license():
    #data = tmp_AES.encrypt("12312312")
    #print(tmp_AES.decrypt(data))
    # 1. 获取UUID
    uuid = get_machine_id()
    license = "4d5ea1060ffc2305b7413ba4bfd035aa"
    # 2. 本地请求数据加密，并给云端验证， 6分钟内 最多请求3次
    encrypt_data, key = regist_license(license, uuid)
    # 3. 云端返回加密结果 其中 key 为random_str iv = '10100202 etimreh'
    if isinstance(key, str):
        key = key.encode('utf-8')
    # 4. 正确的话 返回 {'license': '4d5ea1060ffc2305b7413ba4bfd035aa', 'status': 1, 'license_expire_time': '2020-11-29 15:46:16', 'expire_time': '2020-11-24 15:23:55'}
    # 4. 错误的话 返回错误 原因 不包含status 如"license 不存在"
    result = decode(encrypt_data, key)
    return result


if __name__ == "__main__":
    print(test_license())


