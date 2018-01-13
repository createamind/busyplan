
https://segmentfault.com/a/1190000000578874

server
1. install ss
````
wget --no-check-certificate -O shadowsocks.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks.sh
#or
#curl -o shadowsocks.sh https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks.sh
chmod +x shadowsocks.sh
./shadowsocks.sh 2>&1 | tee shadowsocks.log
````

2.使用指令
启动：/etc/init.d/shadowsocks start

停止：/etc/init.d/shadowsocks stop

重启：/etc/init.d/shadowsocks restart

状态：/etc/init.d/shadowsocks status

3.开启tcp_fastopen
```
echo 3 > /proc/sys/net/ipv4/tcp_fastopen
```
设置开机自动nano /etc/rc.local，在最后增加该行```echo 3 > /proc/sys/net/ipv4/tcp_fastopen```

编辑/etc/sysctl.conf
```
nano /etc/sysctl.conf
```
最后一行增加
```
net.ipv4.tcp_fastopen = 3
```
4.配置文件
nano /etc/shadowsocks.json

```
{
    "server":"0.0.0.0",
    "server_port":6669,
    "local_address":"127.0.0.1",
    "local_port":1080,
    "password":"password",
    "port_password":{
        "6669":"password",
        "6670":"password",
        "6671":"password",
        "6672":"password"
    },
    "timeout":300,
    "method":"aes-256-cfb",
    "fast_open":true
}
```
5.重启
```/etc/init.d/shadowsocks restart```


client
https://github.com/shadowsocks/ShadowsocksX-NG/releases/
