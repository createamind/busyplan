## SSH使用笔记

**远程登录**

```shell
$ ssh user@host

$ ssh -p port user@host //default port 22

```

**口令登录**

可信赖的远程主机的公匙（RSA key fingerprint）保存在/etc/ssh/ssh_known_hosts

**公钥登录**

本地用户输入`$ ssh-keygen`在$HOME/.ssh/目录下生成*公钥*`id_rsa.pub`和*私钥*`id_rsa`l两个文件。

```shell
$ ssh-copy-id user@host

```
上传公钥

**文件传送**

将$HOME/src/目录下面的所有文件，复制到远程主机的$HOME/src/目录。

```shell
　　$ cd && tar czv src | ssh user@host 'tar xz'　
```

将远程主机$HOME/src/目录下面的所有文件，复制到用户的当前目录。

```shell
　　$ ssh user@host 'tar cz src' | tar xzv
```
把home文件打包并解压到/datavg35下

```shell
$ tar -cvf home |(cd /datavg35; tar -xvf -)
```
---;

A机为源服务器,B机为目标服务器
/* 把A机的t11.txt 打包并传送到B机上 */

```shell
[root@erpdemo tmp]# tar cvfz -  t11.txt|ssh erpprd "cat >t11.tar.gz" t11.txt
/* 把A机的t11.txt 打包并传送到B机上的/tmp目录下 */
[root@erpdemo tmp]# tar cvfz  - t11.txt |ssh erpprd "cd /tmp;cat >t11.tar.gz"
t11.txt
/*将A机的t11.txt压缩文件，复制到B机并解压缩*/
[root@erpdemo tmp]#zcat dir.tar | ssh 192.168.0.116 "cd /opt; tar -xf -"
/*在A机上一边打包t11.txt文件，一边传输到B机并解压*/
[root@erpdemo tmp]# tar cvfz - t11.txt |ssh erpprd "cd /tmp; tar -xvfz -"
在补充几个:
传输到远程：tar czf   file| ssh server "tar zxf -"
压缩到远程：tar czf   file| ssh server "cat > file.tar.gz"
解压到远程：ssh server "tar zxf -" < file.tar.gz
解压到本地：ssh server "cat file.tar.gz" | tar zxf -

查看远程主机是否运行进程httpd。

```shell
　　$ ssh user@host 'ps ax | grep [h]ttpd'
```
　　
下面是一个比较有趣的例子。

```
　　$ ssh -L 5900:localhost:5900 host3
```

它表示将本机的5900端口绑定host3的5900端口（这里的localhost指的是host3，因为目标主机是相对host3而言的）。
另一个例子是通过host3的端口转发，ssh登录host2。

```
　　$ ssh -L 9001:host2:22 host3
```
这时，只要ssh登录本机的9001端口，就相当于登录host2了。

```
　　$ ssh -p 9001 localhost
```

上面的-p参数表示指定登录端口。

　　
---

Reference:

* [SSH原理与运用（一）：远程登录](http://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html)
* [SSH原理与运用（二）：远程操作与端口转发](http://www.ruanyifeng.com/blog/2011/12/ssh_port_forwarding.html)
