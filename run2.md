
# 一、服务器端

## 1、执行指令：
```
cargo clean #清除缓存
cargo build --release --features cuda  #有GPU＋cuda，没有去掉

./target/release/spm-cli  --mode worker  --address 0.0.0.0:10128

```
### 2、model地址
```
/// qwen3-vl-8B模型   ,transformers一共36层，需修改topology.yml
/media/nvidia/Elements/Qwen3-vl/Qwen3-VL-8B-Instruct


/// qwen3-vl-2B模型  ,transformers一共28层，需修改topology.yml
/media/nvidia/Elements/Qwen3-vl/Qwen3-VL-2B-Instruct
```

### 3、yml地址
```
/home/nvidia/Spm_llama/topology_qwen3vl.yml
```

# 二、客户端执行指令：

### 1、终端1：（与服务器连通）
```
cargo clean #清除缓存
cargo build --release  #有GPU＋cuda，没有去掉

./target/release/spm-cli --api 0.0.0.0:8082


/// vision-max-side限制图片所占的token数，如果太大的话，首token会很慢
./target/release/spm-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-max-side 512 \
  --vision-no-upscale 
```

### 2、终端2：（大模型问答推理）
```
./target/release/spm-cli  --api-client http://172.16.30.39:8082 --ask "1+1等于多少"

./target/release/spm-cli  --api-client http://127.0.0.1:8082 --image test2.jpg --ask "这张图里有什么？"
```
### 3、model地址
```
/// qwen3-vl-8B模型
/home/firefly/Documents/Qwen3-VL-8B-Instruct

/// qwen3-vl-2B模型
/userdata/Qwen3-VL-2B-Instruct
```
### 4、yml地址
```
/home/firefly/Documents/Spm_llama/topology_qwen3vl.yml
```

# 三、客户端使用NPU执行指令：
### 1、终端1：（与服务器连通）
```
./target/release/spm-cli \
  --mode master \
  --api 0.0.0.0:8082 \
  --vision-rknn /home/firefly/Documents/Spm_llama/vision_448.rknn \
  --vision-fixed-side 448
```
