import bmnetp

def main():
    # please fill in the code
    bmnetp.compile(
        
        model="/workspace/LeNet5/weights/lenet5_mnist.zip",
        shapes=[[1, 1, 32, 32]],
        net_name="lenet5",
        outdir="/workspace/LeNet5/fp32_bmodel",
        target="BM1684",
        opt=2
    )


if __name__ == '__main__':
    main()
