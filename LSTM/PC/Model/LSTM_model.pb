
?
VariableConst*?
value?B?
"???@????????@?1? ??~ ?{??><??>:???????,T???Ǽ?!????#?r???]??vV?????:X>?y????!??a?>vn?>ȍ?=?\???J??C>???????'?>np? <????ݿ1U???l=}?2??8??????Qr???????RS?%&??m??????hMi?R??>????M??9b>?1?>??2????ޑ}>(??>?j??"L??????u>_!!?H???;?=.??>?K??=*?=,8t?????AԿ??Ὅ?O?+?\???*
dtype0
I
Variable/readIdentityVariable*
_class
loc:@Variable*
T0
c

Variable_1Const*A
value8B6
"(??^>??3@C7??O??/??Ĳ<???????>ɮ??!!?>*
dtype0
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
_

Variable_2Const*=
value4B2
"(??n?=o>-??>q?v???G???>??.<??ƾC?7???*
dtype0
O
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0
;

Variable_3Const*
dtype0*
valueB*('$>
O
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0
I
PlaceholderPlaceholder*
dtype0* 
shape:?????????
4
ShapeShapePlaceholder*
T0*
out_type0
A
strided_slice/stackConst*
dtype0*
valueB: 
C
strided_slice/stack_1Const*
valueB:*
dtype0
C
strided_slice/stack_2Const*
valueB:*
dtype0
?
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
ellipsis_mask *
new_axis_mask *

begin_mask *
shrink_axis_mask*
end_mask *
Index0
6
Shape_1ShapePlaceholder*
T0*
out_type0
C
strided_slice_1/stackConst*
dtype0*
valueB:
E
strided_slice_1/stack_1Const*
valueB:*
dtype0
E
strided_slice_1/stack_2Const*
dtype0*
valueB:
?
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
ellipsis_mask *
new_axis_mask *
T0*

begin_mask *
end_mask *
Index0*
shrink_axis_mask
B
Reshape/shapeConst*
dtype0*
valueB"????   
E
ReshapeReshapePlaceholderReshape/shape*
Tshape0*
T0
W
MatMulMatMulReshapeVariable/read*
transpose_a( *
T0*
transpose_b( 
,
addAddMatMulVariable_2/read*
T0
D
Reshape_1/shape/0Const*
valueB :
?????????*
dtype0
;
Reshape_1/shape/2Const*
value	B :
*
dtype0
l
Reshape_1/shapePackReshape_1/shape/0strided_slice_1Reshape_1/shape/2*
T0*

axis *
N
A
	Reshape_1ReshapeaddReshape_1/shape*
T0*
Tshape0
e
;MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims/dimConst*
value	B : *
dtype0
?
7MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims
ExpandDimsstrided_slice;MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0
`
2MultiRNNCellZeroState/BasicLSTMCellZeroState/ConstConst*
dtype0*
valueB:

b
8MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axisConst*
value	B : *
dtype0
?
3MultiRNNCellZeroState/BasicLSTMCellZeroState/concatConcatV27MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims2MultiRNNCellZeroState/BasicLSTMCellZeroState/Const8MultiRNNCellZeroState/BasicLSTMCellZeroState/concat/axis*
T0*
N*

Tidx0
e
8MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0
?
2MultiRNNCellZeroState/BasicLSTMCellZeroState/zerosFill3MultiRNNCellZeroState/BasicLSTMCellZeroState/concat8MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros/Const*
T0
g
=MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
value	B : 
?
9MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_2
ExpandDimsstrided_slice=MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_2/dim*
T0*

Tdim0
b
4MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_2Const*
dtype0*
valueB:

d
:MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axisConst*
dtype0*
value	B : 
?
5MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1ConcatV29MultiRNNCellZeroState/BasicLSTMCellZeroState/ExpandDims_24MultiRNNCellZeroState/BasicLSTMCellZeroState/Const_2:MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1/axis*
T0*

Tidx0*
N
g
:MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0
?
4MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1Fill5MultiRNNCellZeroState/BasicLSTMCellZeroState/concat_1:MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1/Const*
T0
g
=MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims/dimConst*
dtype0*
value	B : 
?
9MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims
ExpandDimsstrided_slice=MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims/dim*

Tdim0*
T0
b
4MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ConstConst*
valueB:
*
dtype0
d
:MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axisConst*
dtype0*
value	B : 
?
5MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concatConcatV29MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims4MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const:MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat/axis*

Tidx0*
N*
T0
g
:MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/ConstConst*
dtype0*
valueB
 *    
?
4MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zerosFill5MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat:MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros/Const*
T0
i
?MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims_2/dimConst*
dtype0*
value	B : 
?
;MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims_2
ExpandDimsstrided_slice?MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims_2/dim*

Tdim0*
T0
d
6MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_2Const*
dtype0*
valueB:

f
<MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0
?
7MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1ConcatV2;MultiRNNCellZeroState/BasicLSTMCellZeroState_1/ExpandDims_26MultiRNNCellZeroState/BasicLSTMCellZeroState_1/Const_2<MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1/axis*
T0*

Tidx0*
N
i
<MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0
?
6MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1Fill7MultiRNNCellZeroState/BasicLSTMCellZeroState_1/concat_1<MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1/Const*
T0
.
RankConst*
dtype0*
value	B :
5
range/startConst*
value	B :*
dtype0
5
range/deltaConst*
dtype0*
value	B :
:
rangeRangerange/startRankrange/delta*

Tidx0
D
concat/values_0Const*
valueB"       *
dtype0
5
concat/axisConst*
dtype0*
value	B : 
U
concatConcatV2concat/values_0rangeconcat/axis*
N*

Tidx0*
T0
?
	transpose	Transpose	Reshape_1concat*
Tperm0*
T0
8
rnn/Shape_1Shape	transpose*
T0*
out_type0
G
rnn/strided_slice_1/stackConst*
dtype0*
valueB: 
I
rnn/strided_slice_1/stack_1Const*
dtype0*
valueB:
I
rnn/strided_slice_1/stack_2Const*
dtype0*
valueB:
?
rnn/strided_slice_1StridedSlicernn/Shape_1rnn/strided_slice_1/stackrnn/strided_slice_1/stack_1rnn/strided_slice_1/stack_2*
ellipsis_mask *
T0*
shrink_axis_mask*
Index0*
end_mask *

begin_mask *
new_axis_mask 
2
rnn/timeConst*
value	B : *
dtype0
?
rnn/TensorArrayTensorArrayV3rnn/strided_slice_1*
dynamic_size( *
clear_after_read(*/
tensor_array_namernn/dynamic_rnn/output_0*
element_shape:*
dtype0
?
rnn/TensorArray_1TensorArrayV3rnn/strided_slice_1*
dynamic_size( *
element_shape:*.
tensor_array_namernn/dynamic_rnn/input_0*
clear_after_read(*
dtype0
I
rnn/TensorArrayUnstack/ShapeShape	transpose*
out_type0*
T0
X
*rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
Z
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
Z
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
?
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
T0*
end_mask *
ellipsis_mask *
shrink_axis_mask*
Index0*

begin_mask *
new_axis_mask 
L
"rnn/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
L
"rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
?
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*

Tidx0
?
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/range	transposernn/TensorArray_1:1*
_class
loc:@transpose*
T0
?
rnn/while/EnterEnterrnn/time*
is_constant( *'

frame_namernn/while/while_context*
T0*
parallel_iterations 
?
rnn/while/Enter_1Enterrnn/TensorArray:1*'

frame_namernn/while/while_context*
is_constant( *
T0*
parallel_iterations 
?
rnn/while/Enter_2Enter2MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros*'

frame_namernn/while/while_context*
T0*
parallel_iterations *
is_constant( 
?
rnn/while/Enter_3Enter4MultiRNNCellZeroState/BasicLSTMCellZeroState/zeros_1*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
T0
?
rnn/while/Enter_4Enter4MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros*'

frame_namernn/while/while_context*
is_constant( *
parallel_iterations *
T0
?
rnn/while/Enter_5Enter6MultiRNNCellZeroState/BasicLSTMCellZeroState_1/zeros_1*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context
T
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
N*
T0
Z
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
N*
T0
Z
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0*
N
Z
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0*
N
Z
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4*
T0*
N
Z
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5*
N*
T0
?
rnn/while/Less/EnterEnterrnn/strided_slice_1*
is_constant(*
T0*
parallel_iterations *'

frame_namernn/while/while_context
F
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0
.
rnn/while/LoopCondLoopCondrnn/while/Less
l
rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*"
_class
loc:@rnn/while/Merge*
T0
r
rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_1*
T0
r
rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_2*
T0
r
rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_3*
T0
r
rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*$
_class
loc:@rnn/while/Merge_4*
T0
r
rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*
T0*$
_class
loc:@rnn/while/Merge_5
;
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0
?
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0
?
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0
?
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0
?
rnn/while/Identity_4Identityrnn/while/Switch_4:1*
T0
?
rnn/while/Identity_5Identityrnn/while/Switch_5:1*
T0
?
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
T0*
is_constant(*'

frame_namernn/while/while_context*
parallel_iterations 
?
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context
?
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity#rnn/while/TensorArrayReadV3/Enter_1*
dtype0
?
0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernelConst*
dtype0*?
value?B?("??i޾?x6>W??>Z?????>"?Z?s '?.??>`?I??o=O?F??>???o;>???>,??>O?F??9??4̤??J??N>l~?>??????=????Qb????&;?͈?h?=}?N?f?H>??=?zL =??=?)????><¦>F~??C??>Bu???I7>?<w??ϝ?-????ͺ?ZWZ?̸'?\?ᾄj >Q???\	????C???꾬?>?mx<"??>??R?仈?Һ???r >>E=7??̲=?4??ޥݼ?{=4)?=6?Ծ??o?z3K??U?>3)=ޮ0>????????????l?D?????u?	?,iѽh??=?Gj>?(?<i?????/<???Ņ*>?i??/?>?`??\ɽZ?ɼ?|"?n<??9?r<x?????>r?[>?$)?r????Ӿa??4g?v??>?C?<֜6> ???ɤ>#d?r????a?9?<>???>?ޯ?)?!>yE̽?s????M?[S>??C?h????ƾ???>1?>?=???=7/j>???G??>??f?\4>'_?=_ݾ?E???=??8?i?=?????@??N$J???)=???F7????վ??о?x???j?<V?þu???8?=?p?qt??R=?6??+??< Ց>??=. >?$M??????>?>h>Ÿ?>&+(>?W??kW??A+9??ϩ{?o=v>y,??H???t?=?R???<?t???ѽ??8???/??"??D?N??	^=??ǽH4>???
>8Z??????????<:X[ƾ??ȼ?>wj>?,?/?`???.>|8?>#??????.>b?:????N?)?f>?;?>1?D?c<???*᥼?8?p?C>+????_N?&?>? ?>?\?7ׅ>?!{??	?<?0??$??n?? Ce??2????=??о0?h??9?<?ۣ?3oD=*??>???>o?>?"?>-??=$/>\?A;R䲼ŦF????>Y>???>=??>ɘ?=??=W4=>3?>?Z =7?:>?=H??/?=jĽ??G>#?B=?z??\?G???Ɠ???ZZ>)7k;j?>??<>?	??+%?z?X???$?P???J??~?=W??>kɌ>q?=n?k>6^?>??)?wY????????P=???@=??Q??????Ӵ?=??|=B?p>??Q>?r??n?k?ƺ?=??>Jr?>X?ݾ??#?v??=?J???u??2I?>?k->p?ͽ??Q?d2?>?ڔ=8R>|9¾??s>}?%=?E??J>???=??۽?M??͚<*??=?$?>??彁c?=?֥??(>??>3???>%?>?w??F??;H???ܡ?>??j???X?L/D??49?J?=Z??=???,??X|??kz>Q???[]?????=?>?9???EN?q
?>[?3?b#?=5?>6?????|>??n???x????????=0\???0??#$?>/??=1_%=8:?=I??>??;>?S?>q[콢8?????!??Fw~?>"?p?????>E?Ǿ?????~?Tmh>??>?*>??4??
?>%?>??p>???>e?d>+Y??y?{?<??? ??>?(=?????i?????ɛ?>??[??=?:??!??=?O?=au??b?>+?????>?0i?F???'*?>rpe???>??>Za6??4?>?U?=N?
?;?T??+?>Hm??I?hi?????0???>gX=>si?=& %?e:?8Ee?9?n?Gւ>Ue?>u?R???>g??>g??>??? 3?=??<????9?>^3?(l農]?>.'??":?=e??>?_l?~?C>!?=??S>?l2???վ??	???c???? >?P<?Wk?????,?????'?>L?R??!>?????R?>???=?f=?|a???@?v??=o?˾?|R???>^?8>?tq>?"%?M??>?5>n?̼pF?>?w??k=????cċ>V?>k??>6P?=$??C?)?W>??>]Db>?mҾ?jɾCm??O'Y?!r?<?;?>[uv?)Υ?z7>??v???g??/>?嬾?]>?þc???%??a??I?0?8?ӽ???aPP>d?????E>4K??Ͳ?0?L>  ??U????>????w?.>?G>r???W?>ڀ????d?i?"??>=3>??j?H"???DX?'?\>5Qü?˞<?u??E?>??????)?<v)?>?>??>?? ?FF?hۀ?C????f=?? ??˾;??>??+????YM+???{?=Z%???qv?M|=MW@>?c?>??=$??O?>.T??Ȁ?HX?=^?C?3?ɼ????+M???P??X>%?>?p?>??׼?->?iQ>\?4?`??>J????.=???>뱶???Ӿ?:滧n'???c???:??_?=2??>?ݾ??S????	??n.??Y?;???A>?k?=2?>F??=??Ƽgހ?	?d:h=??d??????P???????E??>??ֽ?@/??????$>Iŀ>6&????>2????پ??;6?>!?>?l>l]7>q???k>?ǯ??^I??/ >?m{?(Y> ?Ӿ?P??>1?׾?~=???К??:ﾶ?x?^?+?5釾e?)??Ѳ?J?ݾ?????h??M????>m(=? ???!??[?>?N?1?<?x#?˛v?XM뾼??i?D?T{?>??R?¸?yQ6???Ծ?????? ?뾬t???-!????>????\?????=͉??}??2??ۀ??L?>??-=??C?[???Gw>?b??8??=IA1??zC>՗?>?B>?J?=/?N??m;?E???'??sD??:=D>?7Ѿ??,???4=7?I??8*?<n?>=??F??吓?u?>k,?=$????վB?????<x>???=??>?:??=??>??>?uN??t?>&{,?ng???!>$?=>?<J>?H?>?G??L䄓?/?m>??e?(, ?e?{<?έ><?>?J\={??=$q">M ?>?g8???>???>???=??=?m?<?Օ=??>????q??Z?3???8????%?>???,<?F;>i-?l??)??֋>??:???]Ƀ? ??)??kә?C???U?????????????P??x?>c o?$i칏????=??E?>?@????E;??P?w???<X????kľ?!??V+>>???t?w??N?????x??>????
|
5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/readIdentity0rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel*
T0
?
.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biasConst*?
value?B?("??U?m?????>4?,>?c?>?~?>??5?ђ???ѹ>??U????½{??>:?@=?=??y:c??-??<&???<%?PsE?t? ???p?Պ??L??`&?L?0?s??ㇾ??[?Ϊ??V?????+>1??=???>ն?>~??|??#?>????*
dtype0
x
3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/readIdentity.rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias*
T0
?
Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
dtype0
?
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_3Frnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat/axis*
T0*

Tidx0*
N
?
Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/EnterEnter5rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel/read*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
T0
?
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulMatMulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concatGrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
T0*
transpose_a( *
transpose_b( 
?
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/EnterEnter3rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias/read*
T0*
is_constant(*'

frame_namernn/while/while_context*
parallel_iterations 
?
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAddBiasAddArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMulHrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
data_formatNHWC*
T0
?
Jrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimConst^rnn/while/Identity*
dtype0*
value	B :
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/splitSplitJrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split/split_dimBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd*
	num_split*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/yConst^rnn/while/Identity*
dtype0*
valueB
 *  ??
?
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/addAddBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:2@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add/y*
T0
?
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/SigmoidSigmoid>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add*
T0
?
>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mulMulrnn/while/Identity_2Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid*
T0
?
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split*
T0
?
?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/TanhTanhBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:1*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1MulDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_1?rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1Add>rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_1*
T0
?
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Tanh@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0
?
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2SigmoidBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split:3*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2MulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_1Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_2*
T0
?
Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_1/axisConst^rnn/while/Identity*
value	B :*
dtype0
?
Crnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_1ConcatV2@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2rnn/while/Identity_5Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_1/axis*
N*
T0*

Tidx0
?
Crnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_1MatMulCrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/concat_1Grnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul/Enter*
T0*
transpose_a( *
transpose_b( 
?
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_1BiasAddCrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/MatMul_1Hrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC
?
Lrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_1/split_dimConst^rnn/while/Identity*
value	B :*
dtype0
?
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_1SplitLrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_1/split_dimDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/BiasAdd_1*
	num_split*
T0
?
Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_2/yConst^rnn/while/Identity*
valueB
 *  ??*
dtype0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_2AddDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_1:2Brnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_2/y*
T0
?
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_3Sigmoid@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_2*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_3Mulrnn/while/Identity_4Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_3*
T0
?
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_4SigmoidBrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_1*
T0
?
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_2TanhDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_1:1*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_4MulDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_4Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_2*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_3Add@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_3@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_4*
T0
?
Arnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_3Tanh@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_3*
T0
?
Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_5SigmoidDrnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/split_1:3*
T0
?
@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_5MulArnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Tanh_3Drnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/Sigmoid_5*
T0
?
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
is_constant(*
T0*
parallel_iterations *S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_5*'

frame_namernn/while/while_context
?
-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_5rnn/while/Identity_1*
T0*S
_classI
GEloc:@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_5
N
rnn/while/add/yConst^rnn/while/Identity*
value	B :*
dtype0
B
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
T0
@
rnn/while/NextIterationNextIterationrnn/while/add*
T0
b
rnn/while/NextIteration_1NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
u
rnn/while/NextIteration_2NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_1*
T0
u
rnn/while/NextIteration_3NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_2*
T0
u
rnn/while/NextIteration_4NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/add_3*
T0
u
rnn/while/NextIteration_5NextIteration@rnn/while/rnn/multi_rnn_cell/cell_0/cell_0/basic_lstm_cell/mul_5*
T0
5
rnn/while/Exit_1Exitrnn/while/Switch_1*
T0
?
&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_1*"
_class
loc:@rnn/TensorArray
n
 rnn/TensorArrayStack/range/startConst*"
_class
loc:@rnn/TensorArray*
value	B : *
dtype0
n
 rnn/TensorArrayStack/range/deltaConst*
dtype0*
value	B :*"
_class
loc:@rnn/TensorArray
?
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*

Tidx0*"
_class
loc:@rnn/TensorArray
?
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_1*
dtype0*$
element_shape:?????????
*"
_class
loc:@rnn/TensorArray
2
rnn/RankConst*
value	B :*
dtype0
9
rnn/range/startConst*
dtype0*
value	B :
9
rnn/range/deltaConst*
value	B :*
dtype0
J
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*

Tidx0
J
rnn/concat_1/values_0Const*
valueB"       *
dtype0
;
rnn/concat_1/axisConst*
dtype0*
value	B : 
k
rnn/concat_1ConcatV2rnn/concat_1/values_0	rnn/rangernn/concat_1/axis*
N*

Tidx0*
T0
h
rnn/transpose	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_1*
T0*
Tperm0
D
Reshape_2/shapeConst*
dtype0*
valueB"????
   
K
	Reshape_2Reshapernn/transposeReshape_2/shape*
T0*
Tshape0
]
MatMul_1MatMul	Reshape_2Variable_1/read*
T0*
transpose_b( *
transpose_a( 
0
add_1AddMatMul_1Variable_3/read*
T0 