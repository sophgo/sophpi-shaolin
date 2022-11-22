"""
This program is part of the teaching materials for teacher Hao Xiaoli's experimental class of BJTU.

Copyright © 2021 HAO xiaoli and Yang jian.
All rights reserved.
"""

import bmnett
bmnett.compile(model="./LSTM_model.pb",
        outdir="./lstm_out",
        #descs="[0, uint8, 0, 255]",
        shapes=[1, 20,7],
        dyn=False,
        net_name="lstm",
        input_names=["Placeholder"],
        output_names=["add_1"],
        opt=2,
        target="BM1684"
        )



