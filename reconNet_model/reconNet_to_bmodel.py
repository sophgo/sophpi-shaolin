import bmnett


def main():
    bmnett.compile(
        model="./reconNet.pb",
        input_names="x",
        shapes=[[1, 33, 33]],
        #output_names="recon_block/conv2d_6/LeakyRelu/Maximum:0",
        output_names="recon_block/conv2d_5/LeakyRelu",
        net_name="reconNet",
        outdir="./reconNet_bmodel",
        target="BM1684",
        opt=2
    )


if __name__ == '__main__':
    main()
