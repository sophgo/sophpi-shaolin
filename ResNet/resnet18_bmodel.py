import bmnetp

def main():
    # 请补充函数的参数完成模型的编译，请不要修改 outdir 参数，修改可能影响最终评定
    bmnetp.compile(
        model="weights/resnet18_traced.zip",
        outdir="./ResNet18_bmodel",
        shapes=[[1, 3, 224, 224]],
        net_name="resnet18",
        target="BM1684",
        opt=2
    )

if __name__ == '__main__':
    main()

