¼!
æÊ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype

SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.8.02unknown8·ñ
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0

gru_8/gru_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_namegru_8/gru_cell_26/kernel

,gru_8/gru_cell_26/kernel/Read/ReadVariableOpReadVariableOpgru_8/gru_cell_26/kernel*
_output_shapes
:	*
dtype0
¡
"gru_8/gru_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*3
shared_name$"gru_8/gru_cell_26/recurrent_kernel

6gru_8/gru_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru_8/gru_cell_26/recurrent_kernel*
_output_shapes
:	2*
dtype0

gru_8/gru_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_namegru_8/gru_cell_26/bias

*gru_8/gru_cell_26/bias/Read/ReadVariableOpReadVariableOpgru_8/gru_cell_26/bias*
_output_shapes
:	*
dtype0

gru_9/gru_cell_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*)
shared_namegru_9/gru_cell_27/kernel

,gru_9/gru_cell_27/kernel/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_27/kernel*
_output_shapes

:2*
dtype0
 
"gru_9/gru_cell_27/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"gru_9/gru_cell_27/recurrent_kernel

6gru_9/gru_cell_27/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru_9/gru_cell_27/recurrent_kernel*
_output_shapes

:*
dtype0

gru_9/gru_cell_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_namegru_9/gru_cell_27/bias

*gru_9/gru_cell_27/bias/Read/ReadVariableOpReadVariableOpgru_9/gru_cell_27/bias*
_output_shapes

:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

RMSprop/dense_4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_4/kernel/rms

.RMSprop/dense_4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/kernel/rms*
_output_shapes

:*
dtype0

RMSprop/dense_4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_4/bias/rms

,RMSprop/dense_4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/bias/rms*
_output_shapes
:*
dtype0
¥
$RMSprop/gru_8/gru_cell_26/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*5
shared_name&$RMSprop/gru_8/gru_cell_26/kernel/rms

8RMSprop/gru_8/gru_cell_26/kernel/rms/Read/ReadVariableOpReadVariableOp$RMSprop/gru_8/gru_cell_26/kernel/rms*
_output_shapes
:	*
dtype0
¹
.RMSprop/gru_8/gru_cell_26/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*?
shared_name0.RMSprop/gru_8/gru_cell_26/recurrent_kernel/rms
²
BRMSprop/gru_8/gru_cell_26/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp.RMSprop/gru_8/gru_cell_26/recurrent_kernel/rms*
_output_shapes
:	2*
dtype0
¡
"RMSprop/gru_8/gru_cell_26/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*3
shared_name$"RMSprop/gru_8/gru_cell_26/bias/rms

6RMSprop/gru_8/gru_cell_26/bias/rms/Read/ReadVariableOpReadVariableOp"RMSprop/gru_8/gru_cell_26/bias/rms*
_output_shapes
:	*
dtype0
¤
$RMSprop/gru_9/gru_cell_27/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*5
shared_name&$RMSprop/gru_9/gru_cell_27/kernel/rms

8RMSprop/gru_9/gru_cell_27/kernel/rms/Read/ReadVariableOpReadVariableOp$RMSprop/gru_9/gru_cell_27/kernel/rms*
_output_shapes

:2*
dtype0
¸
.RMSprop/gru_9/gru_cell_27/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*?
shared_name0.RMSprop/gru_9/gru_cell_27/recurrent_kernel/rms
±
BRMSprop/gru_9/gru_cell_27/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp.RMSprop/gru_9/gru_cell_27/recurrent_kernel/rms*
_output_shapes

:*
dtype0
 
"RMSprop/gru_9/gru_cell_27/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"RMSprop/gru_9/gru_cell_27/bias/rms

6RMSprop/gru_9/gru_cell_27/bias/rms/Read/ReadVariableOpReadVariableOp"RMSprop/gru_9/gru_cell_27/bias/rms*
_output_shapes

:*
dtype0

NoOpNoOp
¿9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ú8
valueð8Bí8 Bæ8
Û
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
ª
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
ª
cell

state_spec
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*

%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 
¦

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses*

3iter
	4decay
5learning_rate
6momentum
7rho	+rmsz	,rms{	8rms|	9rms}	:rms~	;rms
<rms
=rms*
<
80
91
:2
;3
<4
=5
+6
,7*
<
80
91
:2
;3
<4
=5
+6
,7*
* 
°
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Cserving_default* 
¼

8kernel
9recurrent_kernel
:bias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*
* 

80
91
:2*

80
91
:2*
* 


Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
¼

;kernel
<recurrent_kernel
=bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*
* 

;0
<1
=2*

;0
<1
=2*
* 


[states
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 

fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEgru_8/gru_cell_26/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"gru_8/gru_cell_26/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEgru_8/gru_cell_26/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEgru_9/gru_cell_27/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"gru_9/gru_cell_27/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEgru_9/gru_cell_27/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

k0*
* 
* 
* 

80
91
:2*

80
91
:2*
* 

lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 

;0
<1
=2*

;0
<1
=2*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	vtotal
	wcount
x	variables
y	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

v0
w1*

x	variables*

VARIABLE_VALUERMSprop/dense_4/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUERMSprop/dense_4/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE$RMSprop/gru_8/gru_cell_26/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.RMSprop/gru_8/gru_cell_26/recurrent_kernel/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE"RMSprop/gru_8/gru_cell_26/bias/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUE$RMSprop/gru_9/gru_cell_27/kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE.RMSprop/gru_9/gru_cell_27/recurrent_kernel/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUE"RMSprop/gru_9/gru_cell_27/bias/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_gru_8_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_gru_8_inputgru_8/gru_cell_26/biasgru_8/gru_cell_26/kernel"gru_8/gru_cell_26/recurrent_kernelgru_9/gru_cell_27/biasgru_9/gru_cell_27/kernel"gru_9/gru_cell_27/recurrent_kerneldense_4/kerneldense_4/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_110054
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ö

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp,gru_8/gru_cell_26/kernel/Read/ReadVariableOp6gru_8/gru_cell_26/recurrent_kernel/Read/ReadVariableOp*gru_8/gru_cell_26/bias/Read/ReadVariableOp,gru_9/gru_cell_27/kernel/Read/ReadVariableOp6gru_9/gru_cell_27/recurrent_kernel/Read/ReadVariableOp*gru_9/gru_cell_27/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.RMSprop/dense_4/kernel/rms/Read/ReadVariableOp,RMSprop/dense_4/bias/rms/Read/ReadVariableOp8RMSprop/gru_8/gru_cell_26/kernel/rms/Read/ReadVariableOpBRMSprop/gru_8/gru_cell_26/recurrent_kernel/rms/Read/ReadVariableOp6RMSprop/gru_8/gru_cell_26/bias/rms/Read/ReadVariableOp8RMSprop/gru_9/gru_cell_27/kernel/rms/Read/ReadVariableOpBRMSprop/gru_9/gru_cell_27/recurrent_kernel/rms/Read/ReadVariableOp6RMSprop/gru_9/gru_cell_27/bias/rms/Read/ReadVariableOpConst*$
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_111743

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhogru_8/gru_cell_26/kernel"gru_8/gru_cell_26/recurrent_kernelgru_8/gru_cell_26/biasgru_9/gru_cell_27/kernel"gru_9/gru_cell_27/recurrent_kernelgru_9/gru_cell_27/biastotalcountRMSprop/dense_4/kernel/rmsRMSprop/dense_4/bias/rms$RMSprop/gru_8/gru_cell_26/kernel/rms.RMSprop/gru_8/gru_cell_26/recurrent_kernel/rms"RMSprop/gru_8/gru_cell_26/bias/rms$RMSprop/gru_9/gru_cell_27/kernel/rms.RMSprop/gru_9/gru_cell_27/recurrent_kernel/rms"RMSprop/gru_9/gru_cell_27/bias/rms*#
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_111822ï
M

A__inference_gru_8_layer_call_and_return_conditional_losses_109202

inputs6
#gru_cell_26_readvariableop_resource:	=
*gru_cell_26_matmul_readvariableop_resource:	?
,gru_cell_26_matmul_1_readvariableop_resource:	2
identity¢!gru_cell_26/MatMul/ReadVariableOp¢#gru_cell_26/MatMul_1/ReadVariableOp¢gru_cell_26/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿh
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_cell_26/ReluRelugru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_109113*
condR
while_cond_109112*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2µ
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Û
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_111545

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0
Ú
ª
while_cond_111303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_111303___redundant_placeholder04
0while_while_cond_111303___redundant_placeholder14
0while_while_cond_111303___redundant_placeholder24
0while_while_cond_111303___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
M

A__inference_gru_9_layer_call_and_return_conditional_losses_111240

inputs5
#gru_cell_27_readvariableop_resource:<
*gru_cell_27_matmul_readvariableop_resource:2>
,gru_cell_27_matmul_1_readvariableop_resource:
identity¢!gru_cell_27/MatMul/ReadVariableOp¢#gru_cell_27/MatMul_1/ReadVariableOp¢gru_cell_27/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask~
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿh
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_cell_27/ReluRelugru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_111151*
condR
while_cond_111150*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp"^gru_cell_27/MatMul/ReadVariableOp$^gru_cell_27/MatMul_1/ReadVariableOp^gru_cell_27/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : : 2F
!gru_cell_27/MatMul/ReadVariableOp!gru_cell_27/MatMul/ReadVariableOp2J
#gru_cell_27/MatMul_1/ReadVariableOp#gru_cell_27/MatMul_1/ReadVariableOp28
gru_cell_27/ReadVariableOpgru_cell_27/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
±¯
¸
H__inference_sequential_4_layer_call_and_return_conditional_losses_109707

inputs<
)gru_8_gru_cell_26_readvariableop_resource:	C
0gru_8_gru_cell_26_matmul_readvariableop_resource:	E
2gru_8_gru_cell_26_matmul_1_readvariableop_resource:	2;
)gru_9_gru_cell_27_readvariableop_resource:B
0gru_9_gru_cell_27_matmul_readvariableop_resource:2D
2gru_9_gru_cell_27_matmul_1_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢'gru_8/gru_cell_26/MatMul/ReadVariableOp¢)gru_8/gru_cell_26/MatMul_1/ReadVariableOp¢ gru_8/gru_cell_26/ReadVariableOp¢gru_8/while¢'gru_9/gru_cell_27/MatMul/ReadVariableOp¢)gru_9/gru_cell_27/MatMul_1/ReadVariableOp¢ gru_9/gru_cell_27/ReadVariableOp¢gru_9/whileA
gru_8/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_8/strided_sliceStridedSlicegru_8/Shape:output:0"gru_8/strided_slice/stack:output:0$gru_8/strided_slice/stack_1:output:0$gru_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_8/zeros/packedPackgru_8/strided_slice:output:0gru_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_8/zerosFillgru_8/zeros/packed:output:0gru_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_8/transpose	Transposeinputsgru_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
gru_8/Shape_1Shapegru_8/transpose:y:0*
T0*
_output_shapes
:e
gru_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_8/strided_slice_1StridedSlicegru_8/Shape_1:output:0$gru_8/strided_slice_1/stack:output:0&gru_8/strided_slice_1/stack_1:output:0&gru_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_8/TensorArrayV2TensorListReserve*gru_8/TensorArrayV2/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_8/transpose:y:0Dgru_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_8/strided_slice_2StridedSlicegru_8/transpose:y:0$gru_8/strided_slice_2/stack:output:0&gru_8/strided_slice_2/stack_1:output:0&gru_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
 gru_8/gru_cell_26/ReadVariableOpReadVariableOp)gru_8_gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_8/gru_cell_26/unstackUnpack(gru_8/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'gru_8/gru_cell_26/MatMul/ReadVariableOpReadVariableOp0gru_8_gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¦
gru_8/gru_cell_26/MatMulMatMulgru_8/strided_slice_2:output:0/gru_8/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_8/gru_cell_26/BiasAddBiasAdd"gru_8/gru_cell_26/MatMul:product:0"gru_8/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!gru_8/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_8/gru_cell_26/splitSplit*gru_8/gru_cell_26/split/split_dim:output:0"gru_8/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)gru_8/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp2gru_8_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0 
gru_8/gru_cell_26/MatMul_1MatMulgru_8/zeros:output:01gru_8/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
gru_8/gru_cell_26/BiasAdd_1BiasAdd$gru_8/gru_cell_26/MatMul_1:product:0"gru_8/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru_8/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#gru_8/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_8/gru_cell_26/split_1SplitV$gru_8/gru_cell_26/BiasAdd_1:output:0 gru_8/gru_cell_26/Const:output:0,gru_8/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_8/gru_cell_26/addAddV2 gru_8/gru_cell_26/split:output:0"gru_8/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_8/gru_cell_26/SigmoidSigmoidgru_8/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/add_1AddV2 gru_8/gru_cell_26/split:output:1"gru_8/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
gru_8/gru_cell_26/Sigmoid_1Sigmoidgru_8/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/mulMulgru_8/gru_cell_26/Sigmoid_1:y:0"gru_8/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/add_2AddV2 gru_8/gru_cell_26/split:output:2gru_8/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_8/gru_cell_26/ReluRelugru_8/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/mul_1Mulgru_8/gru_cell_26/Sigmoid:y:0gru_8/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
gru_8/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/gru_cell_26/subSub gru_8/gru_cell_26/sub/x:output:0gru_8/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/mul_2Mulgru_8/gru_cell_26/sub:z:0$gru_8/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/add_3AddV2gru_8/gru_cell_26/mul_1:z:0gru_8/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
#gru_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ê
gru_8/TensorArrayV2_1TensorListReserve,gru_8/TensorArrayV2_1/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_8/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_8/whileWhile!gru_8/while/loop_counter:output:0'gru_8/while/maximum_iterations:output:0gru_8/time:output:0gru_8/TensorArrayV2_1:handle:0gru_8/zeros:output:0gru_8/strided_slice_1:output:0=gru_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_8_gru_cell_26_readvariableop_resource0gru_8_gru_cell_26_matmul_readvariableop_resource2gru_8_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_8_while_body_109461*#
condR
gru_8_while_cond_109460*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
6gru_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ô
(gru_8/TensorArrayV2Stack/TensorListStackTensorListStackgru_8/while:output:3?gru_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0n
gru_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_8/strided_slice_3StridedSlice1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0$gru_8/strided_slice_3/stack:output:0&gru_8/strided_slice_3/stack_1:output:0&gru_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskk
gru_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_8/transpose_1	Transpose1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0gru_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
dropout_8/IdentityIdentitygru_8/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_9/ShapeShapedropout_8/Identity:output:0*
T0*
_output_shapes
:c
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_9/transpose	Transposedropout_8/Identity:output:0gru_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:e
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ò
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
 gru_9/gru_cell_27/ReadVariableOpReadVariableOp)gru_9_gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0
gru_9/gru_cell_27/unstackUnpack(gru_9/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'gru_9/gru_cell_27/MatMul/ReadVariableOpReadVariableOp0gru_9_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0¥
gru_9/gru_cell_27/MatMulMatMulgru_9/strided_slice_2:output:0/gru_9/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/BiasAddBiasAdd"gru_9/gru_cell_27/MatMul:product:0"gru_9/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!gru_9/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_9/gru_cell_27/splitSplit*gru_9/gru_cell_27/split/split_dim:output:0"gru_9/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)gru_9/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp2gru_9_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_9/gru_cell_27/MatMul_1MatMulgru_9/zeros:output:01gru_9/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
gru_9/gru_cell_27/BiasAdd_1BiasAdd$gru_9/gru_cell_27/MatMul_1:product:0"gru_9/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru_9/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#gru_9/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/split_1SplitV$gru_9/gru_cell_27/BiasAdd_1:output:0 gru_9/gru_cell_27/Const:output:0,gru_9/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_9/gru_cell_27/addAddV2 gru_9/gru_cell_27/split:output:0"gru_9/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
gru_9/gru_cell_27/SigmoidSigmoidgru_9/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/add_1AddV2 gru_9/gru_cell_27/split:output:1"gru_9/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_9/gru_cell_27/Sigmoid_1Sigmoidgru_9/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/mulMulgru_9/gru_cell_27/Sigmoid_1:y:0"gru_9/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/add_2AddV2 gru_9/gru_cell_27/split:output:2gru_9/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
gru_9/gru_cell_27/ReluRelugru_9/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/mul_1Mulgru_9/gru_cell_27/Sigmoid:y:0gru_9/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_9/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_9/gru_cell_27/subSub gru_9/gru_cell_27/sub/x:output:0gru_9/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/mul_2Mulgru_9/gru_cell_27/sub:z:0$gru_9/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/add_3AddV2gru_9/gru_cell_27/mul_1:z:0gru_9/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
gru_9/TensorArrayV2_1TensorListReserve,gru_9/TensorArrayV2_1/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_9_gru_cell_27_readvariableop_resource0gru_9_gru_cell_27_matmul_readvariableop_resource2gru_9_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_9_while_body_109611*#
condR
gru_9_while_cond_109610*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ô
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0n
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    p
dropout_9/IdentityIdentitygru_9/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_4/MatMulMatMuldropout_9/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp(^gru_8/gru_cell_26/MatMul/ReadVariableOp*^gru_8/gru_cell_26/MatMul_1/ReadVariableOp!^gru_8/gru_cell_26/ReadVariableOp^gru_8/while(^gru_9/gru_cell_27/MatMul/ReadVariableOp*^gru_9/gru_cell_27/MatMul_1/ReadVariableOp!^gru_9/gru_cell_27/ReadVariableOp^gru_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2R
'gru_8/gru_cell_26/MatMul/ReadVariableOp'gru_8/gru_cell_26/MatMul/ReadVariableOp2V
)gru_8/gru_cell_26/MatMul_1/ReadVariableOp)gru_8/gru_cell_26/MatMul_1/ReadVariableOp2D
 gru_8/gru_cell_26/ReadVariableOp gru_8/gru_cell_26/ReadVariableOp2
gru_8/whilegru_8/while2R
'gru_9/gru_cell_27/MatMul/ReadVariableOp'gru_9/gru_cell_27/MatMul/ReadVariableOp2V
)gru_9/gru_cell_27/MatMul_1/ReadVariableOp)gru_9/gru_cell_27/MatMul_1/ReadVariableOp2D
 gru_9/gru_cell_27/ReadVariableOp gru_9/gru_cell_27/ReadVariableOp2
gru_9/whilegru_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
=

while_body_110162
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_26_readvariableop_resource_0:	E
2while_gru_cell_26_matmul_readvariableop_resource_0:	G
4while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_26_readvariableop_resource:	C
0while_gru_cell_26_matmul_readvariableop_resource:	E
2while_gru_cell_26_matmul_1_readvariableop_resource:	2¢'while/gru_cell_26/MatMul/ReadVariableOp¢)while/gru_cell_26/MatMul_1/ReadVariableOp¢ while/gru_cell_26/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
while/gru_cell_26/ReluReluwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0$while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
ú

H__inference_sequential_4_layer_call_and_return_conditional_losses_109349
gru_8_input
gru_8_109327:	
gru_8_109329:	
gru_8_109331:	2
gru_9_109335:
gru_9_109337:2
gru_9_109339: 
dense_4_109343:
dense_4_109345:
identity¢dense_4/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢gru_8/StatefulPartitionedCall¢gru_9/StatefulPartitionedCall
gru_8/StatefulPartitionedCallStatefulPartitionedCallgru_8_inputgru_8_109327gru_8_109329gru_8_109331*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_109202ñ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_109033
gru_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0gru_9_109335gru_9_109337gru_9_109339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_109004
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_108835
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_4_109343dense_4_109345*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_108779w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^gru_8/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namegru_8_input
=

while_body_108915
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_27_readvariableop_resource_0:D
2while_gru_cell_27_matmul_readvariableop_resource_0:2F
4while_gru_cell_27_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_27_readvariableop_resource:B
0while_gru_cell_27_matmul_readvariableop_resource:2D
2while_gru_cell_27_matmul_1_readvariableop_resource:¢'while/gru_cell_27/MatMul/ReadVariableOp¢)while/gru_cell_27/MatMul_1/ReadVariableOp¢ while/gru_cell_27/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0·
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0 while/gru_cell_27/Const:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
while/gru_cell_27/ReluReluwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0$while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ

while/NoOpNoOp(^while/gru_cell_27/MatMul/ReadVariableOp*^while/gru_cell_27/MatMul_1/ReadVariableOp!^while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_27/MatMul/ReadVariableOp'while/gru_cell_27/MatMul/ReadVariableOp2V
)while/gru_cell_27/MatMul_1/ReadVariableOp)while/gru_cell_27/MatMul_1/ReadVariableOp2D
 while/gru_cell_27/ReadVariableOp while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
 
¯
while_body_108355
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_27_108377_0:,
while_gru_cell_27_108379_0:2,
while_gru_cell_27_108381_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_27_108377:*
while_gru_cell_27_108379:2*
while_gru_cell_27_108381:¢)while/gru_cell_27/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
)while/gru_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_27_108377_0while_gru_cell_27_108379_0while_gru_cell_27_108381_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108303Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/gru_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/gru_cell_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_27_108377while_gru_cell_27_108377_0"6
while_gru_cell_27_108379while_gru_cell_27_108379_0"6
while_gru_cell_27_108381while_gru_cell_27_108381_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/gru_cell_27/StatefulPartitionedCall)while/gru_cell_27/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ô
c
*__inference_dropout_9_layer_call_fn_111403

inputs
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_108835o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
4

A__inference_gru_9_layer_call_and_return_conditional_losses_108419

inputs$
gru_cell_27_108343:$
gru_cell_27_108345:2$
gru_cell_27_108347:
identity¢#gru_cell_27/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskÌ
#gru_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_27_108343gru_cell_27_108345gru_cell_27_108347*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108303n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_27_108343gru_cell_27_108345gru_cell_27_108347*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108355*
condR
while_cond_108354*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^gru_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2J
#gru_cell_27/StatefulPartitionedCall#gru_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
=

while_body_110315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_26_readvariableop_resource_0:	E
2while_gru_cell_26_matmul_readvariableop_resource_0:	G
4while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_26_readvariableop_resource:	C
0while_gru_cell_26_matmul_readvariableop_resource:	E
2while_gru_cell_26_matmul_1_readvariableop_resource:	2¢'while/gru_cell_26/MatMul/ReadVariableOp¢)while/gru_cell_26/MatMul_1/ReadVariableOp¢ while/gru_cell_26/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
while/gru_cell_26/ReluReluwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0$while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
M

A__inference_gru_9_layer_call_and_return_conditional_losses_108754

inputs5
#gru_cell_27_readvariableop_resource:<
*gru_cell_27_matmul_readvariableop_resource:2>
,gru_cell_27_matmul_1_readvariableop_resource:
identity¢!gru_cell_27/MatMul/ReadVariableOp¢#gru_cell_27/MatMul_1/ReadVariableOp¢gru_cell_27/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask~
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿh
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_cell_27/ReluRelugru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108665*
condR
while_cond_108664*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp"^gru_cell_27/MatMul/ReadVariableOp$^gru_cell_27/MatMul_1/ReadVariableOp^gru_cell_27/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : : 2F
!gru_cell_27/MatMul/ReadVariableOp!gru_cell_27/MatMul/ReadVariableOp2J
#gru_cell_27/MatMul_1/ReadVariableOp#gru_cell_27/MatMul_1/ReadVariableOp28
gru_cell_27/ReadVariableOpgru_cell_27/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ó	
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_108835

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
Ö
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108303

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:22
 matmul_1_readvariableop_resource:
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ú
ª
while_cond_108664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108664___redundant_placeholder04
0while_while_cond_108664___redundant_placeholder14
0while_while_cond_108664___redundant_placeholder24
0while_while_cond_108664___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
=

while_body_108498
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_26_readvariableop_resource_0:	E
2while_gru_cell_26_matmul_readvariableop_resource_0:	G
4while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_26_readvariableop_resource:	C
0while_gru_cell_26_matmul_readvariableop_resource:	E
2while_gru_cell_26_matmul_1_readvariableop_resource:	2¢'while/gru_cell_26/MatMul/ReadVariableOp¢)while/gru_cell_26/MatMul_1/ReadVariableOp¢ while/gru_cell_26/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
while/gru_cell_26/ReluReluwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0$while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
´

Ø
,__inference_gru_cell_27_layer_call_fn_111573

inputs
states_0
unknown:
	unknown_0:2
	unknown_1:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108303o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
¢
F
*__inference_dropout_9_layer_call_fn_111398

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_108767`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
ª
while_cond_110997
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110997___redundant_placeholder04
0while_while_cond_110997___redundant_placeholder14
0while_while_cond_110997___redundant_placeholder24
0while_while_cond_110997___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¬
¹
&__inference_gru_8_layer_call_fn_110065
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	2
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_107899|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ú
ª
while_cond_108016
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108016___redundant_placeholder04
0while_while_cond_108016___redundant_placeholder14
0while_while_cond_108016___redundant_placeholder24
0while_while_cond_108016___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
·

Û
,__inference_gru_cell_26_layer_call_fn_111467

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:	2
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107965o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0
4

A__inference_gru_9_layer_call_and_return_conditional_losses_108237

inputs$
gru_cell_27_108161:$
gru_cell_27_108163:2$
gru_cell_27_108165:
identity¢#gru_cell_27/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskÌ
#gru_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_27_108161gru_cell_27_108163gru_cell_27_108165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108160n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_27_108161gru_cell_27_108163gru_cell_27_108165*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108173*
condR
while_cond_108172*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
NoOpNoOp$^gru_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2J
#gru_cell_27/StatefulPartitionedCall#gru_cell_27/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
4

A__inference_gru_8_layer_call_and_return_conditional_losses_107899

inputs%
gru_cell_26_107823:	%
gru_cell_26_107825:	%
gru_cell_26_107827:	2
identity¢#gru_cell_26/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÌ
#gru_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_26_107823gru_cell_26_107825gru_cell_26_107827*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107822n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_26_107823gru_cell_26_107825gru_cell_26_107827*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_107835*
condR
while_cond_107834*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2t
NoOpNoOp$^gru_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#gru_cell_26/StatefulPartitionedCall#gru_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
=

while_body_111151
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_27_readvariableop_resource_0:D
2while_gru_cell_27_matmul_readvariableop_resource_0:2F
4while_gru_cell_27_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_27_readvariableop_resource:B
0while_gru_cell_27_matmul_readvariableop_resource:2D
2while_gru_cell_27_matmul_1_readvariableop_resource:¢'while/gru_cell_27/MatMul/ReadVariableOp¢)while/gru_cell_27/MatMul_1/ReadVariableOp¢ while/gru_cell_27/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0·
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0 while/gru_cell_27/Const:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
while/gru_cell_27/ReluReluwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0$while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ

while/NoOpNoOp(^while/gru_cell_27/MatMul/ReadVariableOp*^while/gru_cell_27/MatMul_1/ReadVariableOp!^while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_27/MatMul/ReadVariableOp'while/gru_cell_27/MatMul/ReadVariableOp2V
)while/gru_cell_27/MatMul_1/ReadVariableOp)while/gru_cell_27/MatMul_1/ReadVariableOp2D
 while/gru_cell_27/ReadVariableOp while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ú
ª
while_cond_110161
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110161___redundant_placeholder04
0while_while_cond_110161___redundant_placeholder14
0while_while_cond_110161___redundant_placeholder24
0while_while_cond_110161___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
è
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_108600

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
M

A__inference_gru_8_layer_call_and_return_conditional_losses_110557

inputs6
#gru_cell_26_readvariableop_resource:	=
*gru_cell_26_matmul_readvariableop_resource:	?
,gru_cell_26_matmul_1_readvariableop_resource:	2
identity¢!gru_cell_26/MatMul/ReadVariableOp¢#gru_cell_26/MatMul_1/ReadVariableOp¢gru_cell_26/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿh
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_cell_26/ReluRelugru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110468*
condR
while_cond_110467*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2µ
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

$sequential_4_gru_8_while_cond_107505B
>sequential_4_gru_8_while_sequential_4_gru_8_while_loop_counterH
Dsequential_4_gru_8_while_sequential_4_gru_8_while_maximum_iterations(
$sequential_4_gru_8_while_placeholder*
&sequential_4_gru_8_while_placeholder_1*
&sequential_4_gru_8_while_placeholder_2D
@sequential_4_gru_8_while_less_sequential_4_gru_8_strided_slice_1Z
Vsequential_4_gru_8_while_sequential_4_gru_8_while_cond_107505___redundant_placeholder0Z
Vsequential_4_gru_8_while_sequential_4_gru_8_while_cond_107505___redundant_placeholder1Z
Vsequential_4_gru_8_while_sequential_4_gru_8_while_cond_107505___redundant_placeholder2Z
Vsequential_4_gru_8_while_sequential_4_gru_8_while_cond_107505___redundant_placeholder3%
!sequential_4_gru_8_while_identity
®
sequential_4/gru_8/while/LessLess$sequential_4_gru_8_while_placeholder@sequential_4_gru_8_while_less_sequential_4_gru_8_strided_slice_1*
T0*
_output_shapes
: q
!sequential_4/gru_8/while/IdentityIdentity!sequential_4/gru_8/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_4_gru_8_while_identity*sequential_4/gru_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
÷
´
&__inference_gru_9_layer_call_fn_110770

inputs
unknown:
	unknown_0:2
	unknown_1:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_108754o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ÑM

A__inference_gru_8_layer_call_and_return_conditional_losses_110251
inputs_06
#gru_cell_26_readvariableop_resource:	=
*gru_cell_26_matmul_readvariableop_resource:	?
,gru_cell_26_matmul_1_readvariableop_resource:	2
identity¢!gru_cell_26/MatMul/ReadVariableOp¢#gru_cell_26/MatMul_1/ReadVariableOp¢gru_cell_26/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿh
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_cell_26/ReluRelugru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110162*
condR
while_cond_110161*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2µ
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

·
&__inference_gru_8_layer_call_fn_110098

inputs
unknown:	
	unknown_0:	
	unknown_1:	2
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_109202s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

·
&__inference_gru_8_layer_call_fn_110087

inputs
unknown:	
	unknown_0:	
	unknown_1:	2
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_108587s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
àD
¼	
gru_8_while_body_109771(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2'
#gru_8_while_gru_8_strided_slice_1_0c
_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0D
1gru_8_while_gru_cell_26_readvariableop_resource_0:	K
8gru_8_while_gru_cell_26_matmul_readvariableop_resource_0:	M
:gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
gru_8_while_identity
gru_8_while_identity_1
gru_8_while_identity_2
gru_8_while_identity_3
gru_8_while_identity_4%
!gru_8_while_gru_8_strided_slice_1a
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensorB
/gru_8_while_gru_cell_26_readvariableop_resource:	I
6gru_8_while_gru_cell_26_matmul_readvariableop_resource:	K
8gru_8_while_gru_cell_26_matmul_1_readvariableop_resource:	2¢-gru_8/while/gru_cell_26/MatMul/ReadVariableOp¢/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp¢&gru_8/while/gru_cell_26/ReadVariableOp
=gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/gru_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0gru_8_while_placeholderFgru_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
&gru_8/while/gru_cell_26/ReadVariableOpReadVariableOp1gru_8_while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_8/while/gru_cell_26/unstackUnpack.gru_8/while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num§
-gru_8/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp8gru_8_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ê
gru_8/while/gru_cell_26/MatMulMatMul6gru_8/while/TensorArrayV2Read/TensorListGetItem:item:05gru_8/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
gru_8/while/gru_cell_26/BiasAddBiasAdd(gru_8/while/gru_cell_26/MatMul:product:0(gru_8/while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'gru_8/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_8/while/gru_cell_26/splitSplit0gru_8/while/gru_cell_26/split/split_dim:output:0(gru_8/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split«
/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp:gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0±
 gru_8/while/gru_cell_26/MatMul_1MatMulgru_8_while_placeholder_27gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
!gru_8/while/gru_cell_26/BiasAdd_1BiasAdd*gru_8/while/gru_cell_26/MatMul_1:product:0(gru_8/while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_8/while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿt
)gru_8/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_8/while/gru_cell_26/split_1SplitV*gru_8/while/gru_cell_26/BiasAdd_1:output:0&gru_8/while/gru_cell_26/Const:output:02gru_8/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split¨
gru_8/while/gru_cell_26/addAddV2&gru_8/while/gru_cell_26/split:output:0(gru_8/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_8/while/gru_cell_26/SigmoidSigmoidgru_8/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ª
gru_8/while/gru_cell_26/add_1AddV2&gru_8/while/gru_cell_26/split:output:1(gru_8/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
!gru_8/while/gru_cell_26/Sigmoid_1Sigmoid!gru_8/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¥
gru_8/while/gru_cell_26/mulMul%gru_8/while/gru_cell_26/Sigmoid_1:y:0(gru_8/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¡
gru_8/while/gru_cell_26/add_2AddV2&gru_8/while/gru_cell_26/split:output:2gru_8/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2y
gru_8/while/gru_cell_26/ReluRelu!gru_8/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/while/gru_cell_26/mul_1Mul#gru_8/while/gru_cell_26/Sigmoid:y:0gru_8_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
gru_8/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_8/while/gru_cell_26/subSub&gru_8/while/gru_cell_26/sub/x:output:0#gru_8/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2£
gru_8/while/gru_cell_26/mul_2Mulgru_8/while/gru_cell_26/sub:z:0*gru_8/while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/while/gru_cell_26/add_3AddV2!gru_8/while/gru_cell_26/mul_1:z:0!gru_8/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ü
0gru_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_8_while_placeholder_1gru_8_while_placeholder!gru_8/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_8/while/addAddV2gru_8_while_placeholdergru_8/while/add/y:output:0*
T0*
_output_shapes
: U
gru_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_8/while/add_1AddV2$gru_8_while_gru_8_while_loop_countergru_8/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_8/while/IdentityIdentitygru_8/while/add_1:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: 
gru_8/while/Identity_1Identity*gru_8_while_gru_8_while_maximum_iterations^gru_8/while/NoOp*
T0*
_output_shapes
: k
gru_8/while/Identity_2Identitygru_8/while/add:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: «
gru_8/while/Identity_3Identity@gru_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_8/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_8/while/Identity_4Identity!gru_8/while/gru_cell_26/add_3:z:0^gru_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ý
gru_8/while/NoOpNoOp.^gru_8/while/gru_cell_26/MatMul/ReadVariableOp0^gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp'^gru_8/while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_8_while_gru_8_strided_slice_1#gru_8_while_gru_8_strided_slice_1_0"v
8gru_8_while_gru_cell_26_matmul_1_readvariableop_resource:gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0"r
6gru_8_while_gru_cell_26_matmul_readvariableop_resource8gru_8_while_gru_cell_26_matmul_readvariableop_resource_0"d
/gru_8_while_gru_cell_26_readvariableop_resource1gru_8_while_gru_cell_26_readvariableop_resource_0"5
gru_8_while_identitygru_8/while/Identity:output:0"9
gru_8_while_identity_1gru_8/while/Identity_1:output:0"9
gru_8_while_identity_2gru_8/while/Identity_2:output:0"9
gru_8_while_identity_3gru_8/while/Identity_3:output:0"9
gru_8_while_identity_4gru_8/while/Identity_4:output:0"À
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2^
-gru_8/while/gru_cell_26/MatMul/ReadVariableOp-gru_8/while/gru_cell_26/MatMul/ReadVariableOp2b
/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp2P
&gru_8/while/gru_cell_26/ReadVariableOp&gru_8/while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
ü

gru_9_while_cond_109927(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1@
<gru_9_while_gru_9_while_cond_109927___redundant_placeholder0@
<gru_9_while_gru_9_while_cond_109927___redundant_placeholder1@
<gru_9_while_gru_9_while_cond_109927___redundant_placeholder2@
<gru_9_while_gru_9_while_cond_109927___redundant_placeholder3
gru_9_while_identity
z
gru_9/while/LessLessgru_9_while_placeholder&gru_9_while_less_gru_9_strided_slice_1*
T0*
_output_shapes
: W
gru_9/while/IdentityIdentitygru_9/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_9_while_identitygru_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ü

$sequential_4_gru_9_while_cond_107655B
>sequential_4_gru_9_while_sequential_4_gru_9_while_loop_counterH
Dsequential_4_gru_9_while_sequential_4_gru_9_while_maximum_iterations(
$sequential_4_gru_9_while_placeholder*
&sequential_4_gru_9_while_placeholder_1*
&sequential_4_gru_9_while_placeholder_2D
@sequential_4_gru_9_while_less_sequential_4_gru_9_strided_slice_1Z
Vsequential_4_gru_9_while_sequential_4_gru_9_while_cond_107655___redundant_placeholder0Z
Vsequential_4_gru_9_while_sequential_4_gru_9_while_cond_107655___redundant_placeholder1Z
Vsequential_4_gru_9_while_sequential_4_gru_9_while_cond_107655___redundant_placeholder2Z
Vsequential_4_gru_9_while_sequential_4_gru_9_while_cond_107655___redundant_placeholder3%
!sequential_4_gru_9_while_identity
®
sequential_4/gru_9/while/LessLess$sequential_4_gru_9_while_placeholder@sequential_4_gru_9_while_less_sequential_4_gru_9_strided_slice_1*
T0*
_output_shapes
: q
!sequential_4/gru_9/while/IdentityIdentity!sequential_4/gru_9/while/Less:z:0*
T0
*
_output_shapes
: "O
!sequential_4_gru_9_while_identity*sequential_4/gru_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

c
*__inference_dropout_8_layer_call_fn_110720

inputs
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_109033s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ222
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
à	
Ë
-__inference_sequential_4_layer_call_fn_109376

inputs
unknown:	
	unknown_0:	
	unknown_1:	2
	unknown_2:
	unknown_3:2
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_108786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
ª
while_cond_107834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_107834___redundant_placeholder04
0while_while_cond_107834___redundant_placeholder14
0while_while_cond_107834___redundant_placeholder24
0while_while_cond_107834___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
Æ	
ô
C__inference_dense_4_layer_call_and_return_conditional_losses_108779

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü

gru_8_while_cond_109460(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2*
&gru_8_while_less_gru_8_strided_slice_1@
<gru_8_while_gru_8_while_cond_109460___redundant_placeholder0@
<gru_8_while_gru_8_while_cond_109460___redundant_placeholder1@
<gru_8_while_gru_8_while_cond_109460___redundant_placeholder2@
<gru_8_while_gru_8_while_cond_109460___redundant_placeholder3
gru_8_while_identity
z
gru_8/while/LessLessgru_8_while_placeholder&gru_8_while_less_gru_8_strided_slice_1*
T0*
_output_shapes
: W
gru_8/while/IdentityIdentitygru_8/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_8_while_identitygru_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
úT
¼
$sequential_4_gru_9_while_body_107656B
>sequential_4_gru_9_while_sequential_4_gru_9_while_loop_counterH
Dsequential_4_gru_9_while_sequential_4_gru_9_while_maximum_iterations(
$sequential_4_gru_9_while_placeholder*
&sequential_4_gru_9_while_placeholder_1*
&sequential_4_gru_9_while_placeholder_2A
=sequential_4_gru_9_while_sequential_4_gru_9_strided_slice_1_0}
ysequential_4_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_9_tensorarrayunstack_tensorlistfromtensor_0P
>sequential_4_gru_9_while_gru_cell_27_readvariableop_resource_0:W
Esequential_4_gru_9_while_gru_cell_27_matmul_readvariableop_resource_0:2Y
Gsequential_4_gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0:%
!sequential_4_gru_9_while_identity'
#sequential_4_gru_9_while_identity_1'
#sequential_4_gru_9_while_identity_2'
#sequential_4_gru_9_while_identity_3'
#sequential_4_gru_9_while_identity_4?
;sequential_4_gru_9_while_sequential_4_gru_9_strided_slice_1{
wsequential_4_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_9_tensorarrayunstack_tensorlistfromtensorN
<sequential_4_gru_9_while_gru_cell_27_readvariableop_resource:U
Csequential_4_gru_9_while_gru_cell_27_matmul_readvariableop_resource:2W
Esequential_4_gru_9_while_gru_cell_27_matmul_1_readvariableop_resource:¢:sequential_4/gru_9/while/gru_cell_27/MatMul/ReadVariableOp¢<sequential_4/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp¢3sequential_4/gru_9/while/gru_cell_27/ReadVariableOp
Jsequential_4/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   
<sequential_4/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_4_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_9_tensorarrayunstack_tensorlistfromtensor_0$sequential_4_gru_9_while_placeholderSsequential_4/gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0²
3sequential_4/gru_9/while/gru_cell_27/ReadVariableOpReadVariableOp>sequential_4_gru_9_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0©
,sequential_4/gru_9/while/gru_cell_27/unstackUnpack;sequential_4/gru_9/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numÀ
:sequential_4/gru_9/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOpEsequential_4_gru_9_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0ð
+sequential_4/gru_9/while/gru_cell_27/MatMulMatMulCsequential_4/gru_9/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_4/gru_9/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
,sequential_4/gru_9/while/gru_cell_27/BiasAddBiasAdd5sequential_4/gru_9/while/gru_cell_27/MatMul:product:05sequential_4/gru_9/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4sequential_4/gru_9/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
*sequential_4/gru_9/while/gru_cell_27/splitSplit=sequential_4/gru_9/while/gru_cell_27/split/split_dim:output:05sequential_4/gru_9/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitÄ
<sequential_4/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOpGsequential_4_gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0×
-sequential_4/gru_9/while/gru_cell_27/MatMul_1MatMul&sequential_4_gru_9_while_placeholder_2Dsequential_4/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
.sequential_4/gru_9/while/gru_cell_27/BiasAdd_1BiasAdd7sequential_4/gru_9/while/gru_cell_27/MatMul_1:product:05sequential_4/gru_9/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_4/gru_9/while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ
6sequential_4/gru_9/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
,sequential_4/gru_9/while/gru_cell_27/split_1SplitV7sequential_4/gru_9/while/gru_cell_27/BiasAdd_1:output:03sequential_4/gru_9/while/gru_cell_27/Const:output:0?sequential_4/gru_9/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitÏ
(sequential_4/gru_9/while/gru_cell_27/addAddV23sequential_4/gru_9/while/gru_cell_27/split:output:05sequential_4/gru_9/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_4/gru_9/while/gru_cell_27/SigmoidSigmoid,sequential_4/gru_9/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
*sequential_4/gru_9/while/gru_cell_27/add_1AddV23sequential_4/gru_9/while/gru_cell_27/split:output:15sequential_4/gru_9/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.sequential_4/gru_9/while/gru_cell_27/Sigmoid_1Sigmoid.sequential_4/gru_9/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ
(sequential_4/gru_9/while/gru_cell_27/mulMul2sequential_4/gru_9/while/gru_cell_27/Sigmoid_1:y:05sequential_4/gru_9/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
*sequential_4/gru_9/while/gru_cell_27/add_2AddV23sequential_4/gru_9/while/gru_cell_27/split:output:2,sequential_4/gru_9/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)sequential_4/gru_9/while/gru_cell_27/ReluRelu.sequential_4/gru_9/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
*sequential_4/gru_9/while/gru_cell_27/mul_1Mul0sequential_4/gru_9/while/gru_cell_27/Sigmoid:y:0&sequential_4_gru_9_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*sequential_4/gru_9/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
(sequential_4/gru_9/while/gru_cell_27/subSub3sequential_4/gru_9/while/gru_cell_27/sub/x:output:00sequential_4/gru_9/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
*sequential_4/gru_9/while/gru_cell_27/mul_2Mul,sequential_4/gru_9/while/gru_cell_27/sub:z:07sequential_4/gru_9/while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
*sequential_4/gru_9/while/gru_cell_27/add_3AddV2.sequential_4/gru_9/while/gru_cell_27/mul_1:z:0.sequential_4/gru_9/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=sequential_4/gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_4_gru_9_while_placeholder_1$sequential_4_gru_9_while_placeholder.sequential_4/gru_9/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_4/gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_4/gru_9/while/addAddV2$sequential_4_gru_9_while_placeholder'sequential_4/gru_9/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_4/gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_4/gru_9/while/add_1AddV2>sequential_4_gru_9_while_sequential_4_gru_9_while_loop_counter)sequential_4/gru_9/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_4/gru_9/while/IdentityIdentity"sequential_4/gru_9/while/add_1:z:0^sequential_4/gru_9/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_4/gru_9/while/Identity_1IdentityDsequential_4_gru_9_while_sequential_4_gru_9_while_maximum_iterations^sequential_4/gru_9/while/NoOp*
T0*
_output_shapes
: 
#sequential_4/gru_9/while/Identity_2Identity sequential_4/gru_9/while/add:z:0^sequential_4/gru_9/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_4/gru_9/while/Identity_3IdentityMsequential_4/gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/gru_9/while/NoOp*
T0*
_output_shapes
: :éèÒ±
#sequential_4/gru_9/while/Identity_4Identity.sequential_4/gru_9/while/gru_cell_27/add_3:z:0^sequential_4/gru_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_4/gru_9/while/NoOpNoOp;^sequential_4/gru_9/while/gru_cell_27/MatMul/ReadVariableOp=^sequential_4/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp4^sequential_4/gru_9/while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Esequential_4_gru_9_while_gru_cell_27_matmul_1_readvariableop_resourceGsequential_4_gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0"
Csequential_4_gru_9_while_gru_cell_27_matmul_readvariableop_resourceEsequential_4_gru_9_while_gru_cell_27_matmul_readvariableop_resource_0"~
<sequential_4_gru_9_while_gru_cell_27_readvariableop_resource>sequential_4_gru_9_while_gru_cell_27_readvariableop_resource_0"O
!sequential_4_gru_9_while_identity*sequential_4/gru_9/while/Identity:output:0"S
#sequential_4_gru_9_while_identity_1,sequential_4/gru_9/while/Identity_1:output:0"S
#sequential_4_gru_9_while_identity_2,sequential_4/gru_9/while/Identity_2:output:0"S
#sequential_4_gru_9_while_identity_3,sequential_4/gru_9/while/Identity_3:output:0"S
#sequential_4_gru_9_while_identity_4,sequential_4/gru_9/while/Identity_4:output:0"|
;sequential_4_gru_9_while_sequential_4_gru_9_strided_slice_1=sequential_4_gru_9_while_sequential_4_gru_9_strided_slice_1_0"ô
wsequential_4_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_9_tensorarrayunstack_tensorlistfromtensorysequential_4_gru_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2x
:sequential_4/gru_9/while/gru_cell_27/MatMul/ReadVariableOp:sequential_4/gru_9/while/gru_cell_27/MatMul/ReadVariableOp2|
<sequential_4/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp<sequential_4/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp2j
3sequential_4/gru_9/while/gru_cell_27/ReadVariableOp3sequential_4/gru_9/while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¿	
Ç
$__inference_signature_wrapper_110054
gru_8_input
unknown:	
	unknown_0:	
	unknown_1:	2
	unknown_2:
	unknown_3:2
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallgru_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_107752o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namegru_8_input
àD
¼	
gru_8_while_body_109461(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2'
#gru_8_while_gru_8_strided_slice_1_0c
_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0D
1gru_8_while_gru_cell_26_readvariableop_resource_0:	K
8gru_8_while_gru_cell_26_matmul_readvariableop_resource_0:	M
:gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
gru_8_while_identity
gru_8_while_identity_1
gru_8_while_identity_2
gru_8_while_identity_3
gru_8_while_identity_4%
!gru_8_while_gru_8_strided_slice_1a
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensorB
/gru_8_while_gru_cell_26_readvariableop_resource:	I
6gru_8_while_gru_cell_26_matmul_readvariableop_resource:	K
8gru_8_while_gru_cell_26_matmul_1_readvariableop_resource:	2¢-gru_8/while/gru_cell_26/MatMul/ReadVariableOp¢/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp¢&gru_8/while/gru_cell_26/ReadVariableOp
=gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ä
/gru_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0gru_8_while_placeholderFgru_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
&gru_8/while/gru_cell_26/ReadVariableOpReadVariableOp1gru_8_while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
gru_8/while/gru_cell_26/unstackUnpack.gru_8/while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num§
-gru_8/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp8gru_8_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Ê
gru_8/while/gru_cell_26/MatMulMatMul6gru_8/while/TensorArrayV2Read/TensorListGetItem:item:05gru_8/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
gru_8/while/gru_cell_26/BiasAddBiasAdd(gru_8/while/gru_cell_26/MatMul:product:0(gru_8/while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'gru_8/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_8/while/gru_cell_26/splitSplit0gru_8/while/gru_cell_26/split/split_dim:output:0(gru_8/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split«
/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp:gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0±
 gru_8/while/gru_cell_26/MatMul_1MatMulgru_8_while_placeholder_27gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
!gru_8/while/gru_cell_26/BiasAdd_1BiasAdd*gru_8/while/gru_cell_26/MatMul_1:product:0(gru_8/while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_8/while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿt
)gru_8/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_8/while/gru_cell_26/split_1SplitV*gru_8/while/gru_cell_26/BiasAdd_1:output:0&gru_8/while/gru_cell_26/Const:output:02gru_8/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split¨
gru_8/while/gru_cell_26/addAddV2&gru_8/while/gru_cell_26/split:output:0(gru_8/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_8/while/gru_cell_26/SigmoidSigmoidgru_8/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2ª
gru_8/while/gru_cell_26/add_1AddV2&gru_8/while/gru_cell_26/split:output:1(gru_8/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
!gru_8/while/gru_cell_26/Sigmoid_1Sigmoid!gru_8/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¥
gru_8/while/gru_cell_26/mulMul%gru_8/while/gru_cell_26/Sigmoid_1:y:0(gru_8/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¡
gru_8/while/gru_cell_26/add_2AddV2&gru_8/while/gru_cell_26/split:output:2gru_8/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2y
gru_8/while/gru_cell_26/ReluRelu!gru_8/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/while/gru_cell_26/mul_1Mul#gru_8/while/gru_cell_26/Sigmoid:y:0gru_8_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
gru_8/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_8/while/gru_cell_26/subSub&gru_8/while/gru_cell_26/sub/x:output:0#gru_8/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2£
gru_8/while/gru_cell_26/mul_2Mulgru_8/while/gru_cell_26/sub:z:0*gru_8/while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/while/gru_cell_26/add_3AddV2!gru_8/while/gru_cell_26/mul_1:z:0!gru_8/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ü
0gru_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_8_while_placeholder_1gru_8_while_placeholder!gru_8/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_8/while/addAddV2gru_8_while_placeholdergru_8/while/add/y:output:0*
T0*
_output_shapes
: U
gru_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_8/while/add_1AddV2$gru_8_while_gru_8_while_loop_countergru_8/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_8/while/IdentityIdentitygru_8/while/add_1:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: 
gru_8/while/Identity_1Identity*gru_8_while_gru_8_while_maximum_iterations^gru_8/while/NoOp*
T0*
_output_shapes
: k
gru_8/while/Identity_2Identitygru_8/while/add:z:0^gru_8/while/NoOp*
T0*
_output_shapes
: «
gru_8/while/Identity_3Identity@gru_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_8/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_8/while/Identity_4Identity!gru_8/while/gru_cell_26/add_3:z:0^gru_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ý
gru_8/while/NoOpNoOp.^gru_8/while/gru_cell_26/MatMul/ReadVariableOp0^gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp'^gru_8/while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_8_while_gru_8_strided_slice_1#gru_8_while_gru_8_strided_slice_1_0"v
8gru_8_while_gru_cell_26_matmul_1_readvariableop_resource:gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0"r
6gru_8_while_gru_cell_26_matmul_readvariableop_resource8gru_8_while_gru_cell_26_matmul_readvariableop_resource_0"d
/gru_8_while_gru_cell_26_readvariableop_resource1gru_8_while_gru_cell_26_readvariableop_resource_0"5
gru_8_while_identitygru_8/while/Identity:output:0"9
gru_8_while_identity_1gru_8/while/Identity_1:output:0"9
gru_8_while_identity_2gru_8/while/Identity_2:output:0"9
gru_8_while_identity_3gru_8/while/Identity_3:output:0"9
gru_8_while_identity_4gru_8/while/Identity_4:output:0"À
]gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_gru_8_while_tensorarrayv2read_tensorlistgetitem_gru_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2^
-gru_8/while/gru_cell_26/MatMul/ReadVariableOp-gru_8/while/gru_cell_26/MatMul/ReadVariableOp2b
/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp2P
&gru_8/while/gru_cell_26/ReadVariableOp&gru_8/while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 


d
E__inference_dropout_8_layer_call_and_return_conditional_losses_109033

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ú
ª
while_cond_110314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110314___redundant_placeholder04
0while_while_cond_110314___redundant_placeholder14
0while_while_cond_110314___redundant_placeholder24
0while_while_cond_110314___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
 
µ
while_body_108017
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_26_108039_0:	-
while_gru_cell_26_108041_0:	-
while_gru_cell_26_108043_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_26_108039:	+
while_gru_cell_26_108041:	+
while_gru_cell_26_108043:	2¢)while/gru_cell_26/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/gru_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_26_108039_0while_gru_cell_26_108041_0while_gru_cell_26_108043_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107965Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/gru_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x

while/NoOpNoOp*^while/gru_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_26_108039while_gru_cell_26_108039_0"6
while_gru_cell_26_108041while_gru_cell_26_108041_0"6
while_gru_cell_26_108043while_gru_cell_26_108043_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2V
)while/gru_cell_26/StatefulPartitionedCall)while/gru_cell_26/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
ÑM

A__inference_gru_8_layer_call_and_return_conditional_losses_110404
inputs_06
#gru_cell_26_readvariableop_resource:	=
*gru_cell_26_matmul_readvariableop_resource:	?
,gru_cell_26_matmul_1_readvariableop_resource:	2
identity¢!gru_cell_26/MatMul/ReadVariableOp¢#gru_cell_26/MatMul_1/ReadVariableOp¢gru_cell_26/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿh
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_cell_26/ReluRelugru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110315*
condR
while_cond_110314*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2µ
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
è
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_110725

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs

Ù
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107822

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates
Ó
	
!__inference__wrapped_model_107752
gru_8_inputI
6sequential_4_gru_8_gru_cell_26_readvariableop_resource:	P
=sequential_4_gru_8_gru_cell_26_matmul_readvariableop_resource:	R
?sequential_4_gru_8_gru_cell_26_matmul_1_readvariableop_resource:	2H
6sequential_4_gru_9_gru_cell_27_readvariableop_resource:O
=sequential_4_gru_9_gru_cell_27_matmul_readvariableop_resource:2Q
?sequential_4_gru_9_gru_cell_27_matmul_1_readvariableop_resource:E
3sequential_4_dense_4_matmul_readvariableop_resource:B
4sequential_4_dense_4_biasadd_readvariableop_resource:
identity¢+sequential_4/dense_4/BiasAdd/ReadVariableOp¢*sequential_4/dense_4/MatMul/ReadVariableOp¢4sequential_4/gru_8/gru_cell_26/MatMul/ReadVariableOp¢6sequential_4/gru_8/gru_cell_26/MatMul_1/ReadVariableOp¢-sequential_4/gru_8/gru_cell_26/ReadVariableOp¢sequential_4/gru_8/while¢4sequential_4/gru_9/gru_cell_27/MatMul/ReadVariableOp¢6sequential_4/gru_9/gru_cell_27/MatMul_1/ReadVariableOp¢-sequential_4/gru_9/gru_cell_27/ReadVariableOp¢sequential_4/gru_9/whileS
sequential_4/gru_8/ShapeShapegru_8_input*
T0*
_output_shapes
:p
&sequential_4/gru_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_4/gru_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_4/gru_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_4/gru_8/strided_sliceStridedSlice!sequential_4/gru_8/Shape:output:0/sequential_4/gru_8/strided_slice/stack:output:01sequential_4/gru_8/strided_slice/stack_1:output:01sequential_4/gru_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_4/gru_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2¬
sequential_4/gru_8/zeros/packedPack)sequential_4/gru_8/strided_slice:output:0*sequential_4/gru_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_4/gru_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_4/gru_8/zerosFill(sequential_4/gru_8/zeros/packed:output:0'sequential_4/gru_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2v
!sequential_4/gru_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_4/gru_8/transpose	Transposegru_8_input*sequential_4/gru_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
sequential_4/gru_8/Shape_1Shape sequential_4/gru_8/transpose:y:0*
T0*
_output_shapes
:r
(sequential_4/gru_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/gru_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/gru_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_4/gru_8/strided_slice_1StridedSlice#sequential_4/gru_8/Shape_1:output:01sequential_4/gru_8/strided_slice_1/stack:output:03sequential_4/gru_8/strided_slice_1/stack_1:output:03sequential_4/gru_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_4/gru_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_4/gru_8/TensorArrayV2TensorListReserve7sequential_4/gru_8/TensorArrayV2/element_shape:output:0+sequential_4/gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_4/gru_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
:sequential_4/gru_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_4/gru_8/transpose:y:0Qsequential_4/gru_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_4/gru_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/gru_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/gru_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_4/gru_8/strided_slice_2StridedSlice sequential_4/gru_8/transpose:y:01sequential_4/gru_8/strided_slice_2/stack:output:03sequential_4/gru_8/strided_slice_2/stack_1:output:03sequential_4/gru_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask¥
-sequential_4/gru_8/gru_cell_26/ReadVariableOpReadVariableOp6sequential_4_gru_8_gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0
&sequential_4/gru_8/gru_cell_26/unstackUnpack5sequential_4/gru_8/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num³
4sequential_4/gru_8/gru_cell_26/MatMul/ReadVariableOpReadVariableOp=sequential_4_gru_8_gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Í
%sequential_4/gru_8/gru_cell_26/MatMulMatMul+sequential_4/gru_8/strided_slice_2:output:0<sequential_4/gru_8/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
&sequential_4/gru_8/gru_cell_26/BiasAddBiasAdd/sequential_4/gru_8/gru_cell_26/MatMul:product:0/sequential_4/gru_8/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.sequential_4/gru_8/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$sequential_4/gru_8/gru_cell_26/splitSplit7sequential_4/gru_8/gru_cell_26/split/split_dim:output:0/sequential_4/gru_8/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split·
6sequential_4/gru_8/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp?sequential_4_gru_8_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0Ç
'sequential_4/gru_8/gru_cell_26/MatMul_1MatMul!sequential_4/gru_8/zeros:output:0>sequential_4/gru_8/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
(sequential_4/gru_8/gru_cell_26/BiasAdd_1BiasAdd1sequential_4/gru_8/gru_cell_26/MatMul_1:product:0/sequential_4/gru_8/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$sequential_4/gru_8/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ{
0sequential_4/gru_8/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&sequential_4/gru_8/gru_cell_26/split_1SplitV1sequential_4/gru_8/gru_cell_26/BiasAdd_1:output:0-sequential_4/gru_8/gru_cell_26/Const:output:09sequential_4/gru_8/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split½
"sequential_4/gru_8/gru_cell_26/addAddV2-sequential_4/gru_8/gru_cell_26/split:output:0/sequential_4/gru_8/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
&sequential_4/gru_8/gru_cell_26/SigmoidSigmoid&sequential_4/gru_8/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¿
$sequential_4/gru_8/gru_cell_26/add_1AddV2-sequential_4/gru_8/gru_cell_26/split:output:1/sequential_4/gru_8/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
(sequential_4/gru_8/gru_cell_26/Sigmoid_1Sigmoid(sequential_4/gru_8/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2º
"sequential_4/gru_8/gru_cell_26/mulMul,sequential_4/gru_8/gru_cell_26/Sigmoid_1:y:0/sequential_4/gru_8/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¶
$sequential_4/gru_8/gru_cell_26/add_2AddV2-sequential_4/gru_8/gru_cell_26/split:output:2&sequential_4/gru_8/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
#sequential_4/gru_8/gru_cell_26/ReluRelu(sequential_4/gru_8/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¬
$sequential_4/gru_8/gru_cell_26/mul_1Mul*sequential_4/gru_8/gru_cell_26/Sigmoid:y:0!sequential_4/gru_8/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
$sequential_4/gru_8/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"sequential_4/gru_8/gru_cell_26/subSub-sequential_4/gru_8/gru_cell_26/sub/x:output:0*sequential_4/gru_8/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2¸
$sequential_4/gru_8/gru_cell_26/mul_2Mul&sequential_4/gru_8/gru_cell_26/sub:z:01sequential_4/gru_8/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2³
$sequential_4/gru_8/gru_cell_26/add_3AddV2(sequential_4/gru_8/gru_cell_26/mul_1:z:0(sequential_4/gru_8/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
0sequential_4/gru_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ñ
"sequential_4/gru_8/TensorArrayV2_1TensorListReserve9sequential_4/gru_8/TensorArrayV2_1/element_shape:output:0+sequential_4/gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_4/gru_8/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_4/gru_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_4/gru_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : µ
sequential_4/gru_8/whileWhile.sequential_4/gru_8/while/loop_counter:output:04sequential_4/gru_8/while/maximum_iterations:output:0 sequential_4/gru_8/time:output:0+sequential_4/gru_8/TensorArrayV2_1:handle:0!sequential_4/gru_8/zeros:output:0+sequential_4/gru_8/strided_slice_1:output:0Jsequential_4/gru_8/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_4_gru_8_gru_cell_26_readvariableop_resource=sequential_4_gru_8_gru_cell_26_matmul_readvariableop_resource?sequential_4_gru_8_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$sequential_4_gru_8_while_body_107506*0
cond(R&
$sequential_4_gru_8_while_cond_107505*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
Csequential_4/gru_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   û
5sequential_4/gru_8/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_4/gru_8/while:output:3Lsequential_4/gru_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0{
(sequential_4/gru_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_4/gru_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/gru_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_4/gru_8/strided_slice_3StridedSlice>sequential_4/gru_8/TensorArrayV2Stack/TensorListStack:tensor:01sequential_4/gru_8/strided_slice_3/stack:output:03sequential_4/gru_8/strided_slice_3/stack_1:output:03sequential_4/gru_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskx
#sequential_4/gru_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_4/gru_8/transpose_1	Transpose>sequential_4/gru_8/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_4/gru_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
sequential_4/gru_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_4/dropout_8/IdentityIdentity"sequential_4/gru_8/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2p
sequential_4/gru_9/ShapeShape(sequential_4/dropout_8/Identity:output:0*
T0*
_output_shapes
:p
&sequential_4/gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential_4/gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential_4/gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sequential_4/gru_9/strided_sliceStridedSlice!sequential_4/gru_9/Shape:output:0/sequential_4/gru_9/strided_slice/stack:output:01sequential_4/gru_9/strided_slice/stack_1:output:01sequential_4/gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!sequential_4/gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¬
sequential_4/gru_9/zeros/packedPack)sequential_4/gru_9/strided_slice:output:0*sequential_4/gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:c
sequential_4/gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¥
sequential_4/gru_9/zerosFill(sequential_4/gru_9/zeros/packed:output:0'sequential_4/gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
!sequential_4/gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          µ
sequential_4/gru_9/transpose	Transpose(sequential_4/dropout_8/Identity:output:0*sequential_4/gru_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
sequential_4/gru_9/Shape_1Shape sequential_4/gru_9/transpose:y:0*
T0*
_output_shapes
:r
(sequential_4/gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:º
"sequential_4/gru_9/strided_slice_1StridedSlice#sequential_4/gru_9/Shape_1:output:01sequential_4/gru_9/strided_slice_1/stack:output:03sequential_4/gru_9/strided_slice_1/stack_1:output:03sequential_4/gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.sequential_4/gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿí
 sequential_4/gru_9/TensorArrayV2TensorListReserve7sequential_4/gru_9/TensorArrayV2/element_shape:output:0+sequential_4/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Hsequential_4/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   
:sequential_4/gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor sequential_4/gru_9/transpose:y:0Qsequential_4/gru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒr
(sequential_4/gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*sequential_4/gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
"sequential_4/gru_9/strided_slice_2StridedSlice sequential_4/gru_9/transpose:y:01sequential_4/gru_9/strided_slice_2/stack:output:03sequential_4/gru_9/strided_slice_2/stack_1:output:03sequential_4/gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask¤
-sequential_4/gru_9/gru_cell_27/ReadVariableOpReadVariableOp6sequential_4_gru_9_gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0
&sequential_4/gru_9/gru_cell_27/unstackUnpack5sequential_4/gru_9/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num²
4sequential_4/gru_9/gru_cell_27/MatMul/ReadVariableOpReadVariableOp=sequential_4_gru_9_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0Ì
%sequential_4/gru_9/gru_cell_27/MatMulMatMul+sequential_4/gru_9/strided_slice_2:output:0<sequential_4/gru_9/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
&sequential_4/gru_9/gru_cell_27/BiasAddBiasAdd/sequential_4/gru_9/gru_cell_27/MatMul:product:0/sequential_4/gru_9/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
.sequential_4/gru_9/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
$sequential_4/gru_9/gru_cell_27/splitSplit7sequential_4/gru_9/gru_cell_27/split/split_dim:output:0/sequential_4/gru_9/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¶
6sequential_4/gru_9/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp?sequential_4_gru_9_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0Æ
'sequential_4/gru_9/gru_cell_27/MatMul_1MatMul!sequential_4/gru_9/zeros:output:0>sequential_4/gru_9/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
(sequential_4/gru_9/gru_cell_27/BiasAdd_1BiasAdd1sequential_4/gru_9/gru_cell_27/MatMul_1:product:0/sequential_4/gru_9/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$sequential_4/gru_9/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ{
0sequential_4/gru_9/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
&sequential_4/gru_9/gru_cell_27/split_1SplitV1sequential_4/gru_9/gru_cell_27/BiasAdd_1:output:0-sequential_4/gru_9/gru_cell_27/Const:output:09sequential_4/gru_9/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split½
"sequential_4/gru_9/gru_cell_27/addAddV2-sequential_4/gru_9/gru_cell_27/split:output:0/sequential_4/gru_9/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&sequential_4/gru_9/gru_cell_27/SigmoidSigmoid&sequential_4/gru_9/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
$sequential_4/gru_9/gru_cell_27/add_1AddV2-sequential_4/gru_9/gru_cell_27/split:output:1/sequential_4/gru_9/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(sequential_4/gru_9/gru_cell_27/Sigmoid_1Sigmoid(sequential_4/gru_9/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"sequential_4/gru_9/gru_cell_27/mulMul,sequential_4/gru_9/gru_cell_27/Sigmoid_1:y:0/sequential_4/gru_9/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
$sequential_4/gru_9/gru_cell_27/add_2AddV2-sequential_4/gru_9/gru_cell_27/split:output:2&sequential_4/gru_9/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#sequential_4/gru_9/gru_cell_27/ReluRelu(sequential_4/gru_9/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
$sequential_4/gru_9/gru_cell_27/mul_1Mul*sequential_4/gru_9/gru_cell_27/Sigmoid:y:0!sequential_4/gru_9/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$sequential_4/gru_9/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
"sequential_4/gru_9/gru_cell_27/subSub-sequential_4/gru_9/gru_cell_27/sub/x:output:0*sequential_4/gru_9/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
$sequential_4/gru_9/gru_cell_27/mul_2Mul&sequential_4/gru_9/gru_cell_27/sub:z:01sequential_4/gru_9/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
$sequential_4/gru_9/gru_cell_27/add_3AddV2(sequential_4/gru_9/gru_cell_27/mul_1:z:0(sequential_4/gru_9/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0sequential_4/gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ñ
"sequential_4/gru_9/TensorArrayV2_1TensorListReserve9sequential_4/gru_9/TensorArrayV2_1/element_shape:output:0+sequential_4/gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒY
sequential_4/gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+sequential_4/gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿg
%sequential_4/gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : µ
sequential_4/gru_9/whileWhile.sequential_4/gru_9/while/loop_counter:output:04sequential_4/gru_9/while/maximum_iterations:output:0 sequential_4/gru_9/time:output:0+sequential_4/gru_9/TensorArrayV2_1:handle:0!sequential_4/gru_9/zeros:output:0+sequential_4/gru_9/strided_slice_1:output:0Jsequential_4/gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:06sequential_4_gru_9_gru_cell_27_readvariableop_resource=sequential_4_gru_9_gru_cell_27_matmul_readvariableop_resource?sequential_4_gru_9_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *0
body(R&
$sequential_4_gru_9_while_body_107656*0
cond(R&
$sequential_4_gru_9_while_cond_107655*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Csequential_4/gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   û
5sequential_4/gru_9/TensorArrayV2Stack/TensorListStackTensorListStack!sequential_4/gru_9/while:output:3Lsequential_4/gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0{
(sequential_4/gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿt
*sequential_4/gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*sequential_4/gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:æ
"sequential_4/gru_9/strided_slice_3StridedSlice>sequential_4/gru_9/TensorArrayV2Stack/TensorListStack:tensor:01sequential_4/gru_9/strided_slice_3/stack:output:03sequential_4/gru_9/strided_slice_3/stack_1:output:03sequential_4/gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskx
#sequential_4/gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ï
sequential_4/gru_9/transpose_1	Transpose>sequential_4/gru_9/TensorArrayV2Stack/TensorListStack:tensor:0,sequential_4/gru_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
sequential_4/gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
sequential_4/dropout_9/IdentityIdentity+sequential_4/gru_9/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0µ
sequential_4/dense_4/MatMulMatMul(sequential_4/dropout_9/Identity:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%sequential_4/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp5^sequential_4/gru_8/gru_cell_26/MatMul/ReadVariableOp7^sequential_4/gru_8/gru_cell_26/MatMul_1/ReadVariableOp.^sequential_4/gru_8/gru_cell_26/ReadVariableOp^sequential_4/gru_8/while5^sequential_4/gru_9/gru_cell_27/MatMul/ReadVariableOp7^sequential_4/gru_9/gru_cell_27/MatMul_1/ReadVariableOp.^sequential_4/gru_9/gru_cell_27/ReadVariableOp^sequential_4/gru_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2l
4sequential_4/gru_8/gru_cell_26/MatMul/ReadVariableOp4sequential_4/gru_8/gru_cell_26/MatMul/ReadVariableOp2p
6sequential_4/gru_8/gru_cell_26/MatMul_1/ReadVariableOp6sequential_4/gru_8/gru_cell_26/MatMul_1/ReadVariableOp2^
-sequential_4/gru_8/gru_cell_26/ReadVariableOp-sequential_4/gru_8/gru_cell_26/ReadVariableOp24
sequential_4/gru_8/whilesequential_4/gru_8/while2l
4sequential_4/gru_9/gru_cell_27/MatMul/ReadVariableOp4sequential_4/gru_9/gru_cell_27/MatMul/ReadVariableOp2p
6sequential_4/gru_9/gru_cell_27/MatMul_1/ReadVariableOp6sequential_4/gru_9/gru_cell_27/MatMul_1/ReadVariableOp2^
-sequential_4/gru_9/gru_cell_27/ReadVariableOp-sequential_4/gru_9/gru_cell_27/ReadVariableOp24
sequential_4/gru_9/whilesequential_4/gru_9/while:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namegru_8_input
´

Ø
,__inference_gru_cell_27_layer_call_fn_111559

inputs
states_0
unknown:
	unknown_0:2
	unknown_1:
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
Ã

(__inference_dense_4_layer_call_fn_111429

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_108779o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÑD
¶	
gru_9_while_body_109928(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2'
#gru_9_while_gru_9_strided_slice_1_0c
_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0C
1gru_9_while_gru_cell_27_readvariableop_resource_0:J
8gru_9_while_gru_cell_27_matmul_readvariableop_resource_0:2L
:gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0:
gru_9_while_identity
gru_9_while_identity_1
gru_9_while_identity_2
gru_9_while_identity_3
gru_9_while_identity_4%
!gru_9_while_gru_9_strided_slice_1a
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensorA
/gru_9_while_gru_cell_27_readvariableop_resource:H
6gru_9_while_gru_cell_27_matmul_readvariableop_resource:2J
8gru_9_while_gru_cell_27_matmul_1_readvariableop_resource:¢-gru_9/while/gru_cell_27/MatMul/ReadVariableOp¢/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp¢&gru_9/while/gru_cell_27/ReadVariableOp
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ä
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
&gru_9/while/gru_cell_27/ReadVariableOpReadVariableOp1gru_9_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
gru_9/while/gru_cell_27/unstackUnpack.gru_9/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
-gru_9/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp8gru_9_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0É
gru_9/while/gru_cell_27/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:05gru_9/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
gru_9/while/gru_cell_27/BiasAddBiasAdd(gru_9/while/gru_cell_27/MatMul:product:0(gru_9/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'gru_9/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_9/while/gru_cell_27/splitSplit0gru_9/while/gru_cell_27/split/split_dim:output:0(gru_9/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitª
/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp:gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0°
 gru_9/while/gru_cell_27/MatMul_1MatMulgru_9_while_placeholder_27gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!gru_9/while/gru_cell_27/BiasAdd_1BiasAdd*gru_9/while/gru_cell_27/MatMul_1:product:0(gru_9/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_9/while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿt
)gru_9/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_9/while/gru_cell_27/split_1SplitV*gru_9/while/gru_cell_27/BiasAdd_1:output:0&gru_9/while/gru_cell_27/Const:output:02gru_9/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¨
gru_9/while/gru_cell_27/addAddV2&gru_9/while/gru_cell_27/split:output:0(gru_9/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_9/while/gru_cell_27/SigmoidSigmoidgru_9/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
gru_9/while/gru_cell_27/add_1AddV2&gru_9/while/gru_cell_27/split:output:1(gru_9/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!gru_9/while/gru_cell_27/Sigmoid_1Sigmoid!gru_9/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
gru_9/while/gru_cell_27/mulMul%gru_9/while/gru_cell_27/Sigmoid_1:y:0(gru_9/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
gru_9/while/gru_cell_27/add_2AddV2&gru_9/while/gru_cell_27/split:output:2gru_9/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_9/while/gru_cell_27/ReluRelu!gru_9/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/while/gru_cell_27/mul_1Mul#gru_9/while/gru_cell_27/Sigmoid:y:0gru_9_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
gru_9/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_9/while/gru_cell_27/subSub&gru_9/while/gru_cell_27/sub/x:output:0#gru_9/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
gru_9/while/gru_cell_27/mul_2Mulgru_9/while/gru_cell_27/sub:z:0*gru_9/while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/while/gru_cell_27/add_3AddV2!gru_9/while/gru_cell_27/mul_1:z:0!gru_9/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
0gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_9_while_placeholder_1gru_9_while_placeholder!gru_9/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_9/while/addAddV2gru_9_while_placeholdergru_9/while/add/y:output:0*
T0*
_output_shapes
: U
gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations^gru_9/while/NoOp*
T0*
_output_shapes
: k
gru_9/while/Identity_2Identitygru_9/while/add:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: «
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_9/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_9/while/Identity_4Identity!gru_9/while/gru_cell_27/add_3:z:0^gru_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
gru_9/while/NoOpNoOp.^gru_9/while/gru_cell_27/MatMul/ReadVariableOp0^gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp'^gru_9/while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"v
8gru_9_while_gru_cell_27_matmul_1_readvariableop_resource:gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0"r
6gru_9_while_gru_cell_27_matmul_readvariableop_resource8gru_9_while_gru_cell_27_matmul_readvariableop_resource_0"d
/gru_9_while_gru_cell_27_readvariableop_resource1gru_9_while_gru_cell_27_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"À
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2^
-gru_9/while/gru_cell_27/MatMul/ReadVariableOp-gru_9/while/gru_cell_27/MatMul/ReadVariableOp2b
/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp2P
&gru_9/while/gru_cell_27/ReadVariableOp&gru_9/while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
 
µ
while_body_107835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_gru_cell_26_107857_0:	-
while_gru_cell_26_107859_0:	-
while_gru_cell_26_107861_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_gru_cell_26_107857:	+
while_gru_cell_26_107859:	+
while_gru_cell_26_107861:	2¢)while/gru_cell_26/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
)while/gru_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_26_107857_0while_gru_cell_26_107859_0while_gru_cell_26_107861_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107822Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_26/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/gru_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2x

while/NoOpNoOp*^while/gru_cell_26/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_26_107857while_gru_cell_26_107857_0"6
while_gru_cell_26_107859while_gru_cell_26_107859_0"6
while_gru_cell_26_107861while_gru_cell_26_107861_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2V
)while/gru_cell_26/StatefulPartitionedCall)while/gru_cell_26/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
ü

gru_8_while_cond_109770(
$gru_8_while_gru_8_while_loop_counter.
*gru_8_while_gru_8_while_maximum_iterations
gru_8_while_placeholder
gru_8_while_placeholder_1
gru_8_while_placeholder_2*
&gru_8_while_less_gru_8_strided_slice_1@
<gru_8_while_gru_8_while_cond_109770___redundant_placeholder0@
<gru_8_while_gru_8_while_cond_109770___redundant_placeholder1@
<gru_8_while_gru_8_while_cond_109770___redundant_placeholder2@
<gru_8_while_gru_8_while_cond_109770___redundant_placeholder3
gru_8_while_identity
z
gru_8/while/LessLessgru_8_while_placeholder&gru_8_while_less_gru_8_strided_slice_1*
T0*
_output_shapes
: W
gru_8/while/IdentityIdentitygru_8/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_8_while_identitygru_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
4

A__inference_gru_8_layer_call_and_return_conditional_losses_108081

inputs%
gru_cell_26_108005:	%
gru_cell_26_108007:	%
gru_cell_26_108009:	2
identity¢#gru_cell_26/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskÌ
#gru_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_26_108005gru_cell_26_108007gru_cell_26_108009*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107965n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : û
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_26_108005gru_cell_26_108007gru_cell_26_108009*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108017*
condR
while_cond_108016*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2t
NoOpNoOp$^gru_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#gru_cell_26/StatefulPartitionedCall#gru_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ¾
¸
H__inference_sequential_4_layer_call_and_return_conditional_losses_110031

inputs<
)gru_8_gru_cell_26_readvariableop_resource:	C
0gru_8_gru_cell_26_matmul_readvariableop_resource:	E
2gru_8_gru_cell_26_matmul_1_readvariableop_resource:	2;
)gru_9_gru_cell_27_readvariableop_resource:B
0gru_9_gru_cell_27_matmul_readvariableop_resource:2D
2gru_9_gru_cell_27_matmul_1_readvariableop_resource:8
&dense_4_matmul_readvariableop_resource:5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢'gru_8/gru_cell_26/MatMul/ReadVariableOp¢)gru_8/gru_cell_26/MatMul_1/ReadVariableOp¢ gru_8/gru_cell_26/ReadVariableOp¢gru_8/while¢'gru_9/gru_cell_27/MatMul/ReadVariableOp¢)gru_9/gru_cell_27/MatMul_1/ReadVariableOp¢ gru_9/gru_cell_27/ReadVariableOp¢gru_9/whileA
gru_8/ShapeShapeinputs*
T0*
_output_shapes
:c
gru_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_8/strided_sliceStridedSlicegru_8/Shape:output:0"gru_8/strided_slice/stack:output:0$gru_8/strided_slice/stack_1:output:0$gru_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
gru_8/zeros/packedPackgru_8/strided_slice:output:0gru_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_8/zerosFillgru_8/zeros/packed:output:0gru_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
gru_8/transpose	Transposeinputsgru_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
gru_8/Shape_1Shapegru_8/transpose:y:0*
T0*
_output_shapes
:e
gru_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_8/strided_slice_1StridedSlicegru_8/Shape_1:output:0$gru_8/strided_slice_1/stack:output:0&gru_8/strided_slice_1/stack_1:output:0&gru_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_8/TensorArrayV2TensorListReserve*gru_8/TensorArrayV2/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ò
-gru_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_8/transpose:y:0Dgru_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_8/strided_slice_2StridedSlicegru_8/transpose:y:0$gru_8/strided_slice_2/stack:output:0&gru_8/strided_slice_2/stack_1:output:0&gru_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
 gru_8/gru_cell_26/ReadVariableOpReadVariableOp)gru_8_gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_8/gru_cell_26/unstackUnpack(gru_8/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'gru_8/gru_cell_26/MatMul/ReadVariableOpReadVariableOp0gru_8_gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0¦
gru_8/gru_cell_26/MatMulMatMulgru_8/strided_slice_2:output:0/gru_8/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_8/gru_cell_26/BiasAddBiasAdd"gru_8/gru_cell_26/MatMul:product:0"gru_8/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!gru_8/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_8/gru_cell_26/splitSplit*gru_8/gru_cell_26/split/split_dim:output:0"gru_8/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)gru_8/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp2gru_8_gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0 
gru_8/gru_cell_26/MatMul_1MatMulgru_8/zeros:output:01gru_8/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
gru_8/gru_cell_26/BiasAdd_1BiasAdd$gru_8/gru_cell_26/MatMul_1:product:0"gru_8/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru_8/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#gru_8/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_8/gru_cell_26/split_1SplitV$gru_8/gru_cell_26/BiasAdd_1:output:0 gru_8/gru_cell_26/Const:output:0,gru_8/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_8/gru_cell_26/addAddV2 gru_8/gru_cell_26/split:output:0"gru_8/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
gru_8/gru_cell_26/SigmoidSigmoidgru_8/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/add_1AddV2 gru_8/gru_cell_26/split:output:1"gru_8/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
gru_8/gru_cell_26/Sigmoid_1Sigmoidgru_8/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/mulMulgru_8/gru_cell_26/Sigmoid_1:y:0"gru_8/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/add_2AddV2 gru_8/gru_cell_26/split:output:2gru_8/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
gru_8/gru_cell_26/ReluRelugru_8/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/mul_1Mulgru_8/gru_cell_26/Sigmoid:y:0gru_8/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
gru_8/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_8/gru_cell_26/subSub gru_8/gru_cell_26/sub/x:output:0gru_8/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/mul_2Mulgru_8/gru_cell_26/sub:z:0$gru_8/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_8/gru_cell_26/add_3AddV2gru_8/gru_cell_26/mul_1:z:0gru_8/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2t
#gru_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ê
gru_8/TensorArrayV2_1TensorListReserve,gru_8/TensorArrayV2_1/element_shape:output:0gru_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_8/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_8/whileWhile!gru_8/while/loop_counter:output:0'gru_8/while/maximum_iterations:output:0gru_8/time:output:0gru_8/TensorArrayV2_1:handle:0gru_8/zeros:output:0gru_8/strided_slice_1:output:0=gru_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_8_gru_cell_26_readvariableop_resource0gru_8_gru_cell_26_matmul_readvariableop_resource2gru_8_gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_8_while_body_109771*#
condR
gru_8_while_cond_109770*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
6gru_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ô
(gru_8/TensorArrayV2Stack/TensorListStackTensorListStackgru_8/while:output:3?gru_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0n
gru_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_8/strided_slice_3StridedSlice1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0$gru_8/strided_slice_3/stack:output:0&gru_8/strided_slice_3/stack_1:output:0&gru_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maskk
gru_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_8/transpose_1	Transpose1gru_8/TensorArrayV2Stack/TensorListStack:tensor:0gru_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_8/dropout/MulMulgru_8/transpose_1:y:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
dropout_8/dropout/ShapeShapegru_8/transpose_1:y:0*
T0*
_output_shapes
:¤
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>È
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_9/ShapeShapedropout_8/dropout/Mul_1:z:0*
T0*
_output_shapes
:c
gru_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
gru_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
gru_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
gru_9/strided_sliceStridedSlicegru_9/Shape:output:0"gru_9/strided_slice/stack:output:0$gru_9/strided_slice/stack_1:output:0$gru_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskV
gru_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
gru_9/zeros/packedPackgru_9/strided_slice:output:0gru_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:V
gru_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
gru_9/zerosFillgru_9/zeros/packed:output:0gru_9/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
gru_9/transpose	Transposedropout_8/dropout/Mul_1:z:0gru_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2P
gru_9/Shape_1Shapegru_9/transpose:y:0*
T0*
_output_shapes
:e
gru_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
gru_9/strided_slice_1StridedSlicegru_9/Shape_1:output:0$gru_9/strided_slice_1/stack:output:0&gru_9/strided_slice_1/stack_1:output:0&gru_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
!gru_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
gru_9/TensorArrayV2TensorListReserve*gru_9/TensorArrayV2/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
;gru_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ò
-gru_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru_9/transpose:y:0Dgru_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒe
gru_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
gru_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
gru_9/strided_slice_2StridedSlicegru_9/transpose:y:0$gru_9/strided_slice_2/stack:output:0&gru_9/strided_slice_2/stack_1:output:0&gru_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask
 gru_9/gru_cell_27/ReadVariableOpReadVariableOp)gru_9_gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0
gru_9/gru_cell_27/unstackUnpack(gru_9/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'gru_9/gru_cell_27/MatMul/ReadVariableOpReadVariableOp0gru_9_gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0¥
gru_9/gru_cell_27/MatMulMatMulgru_9/strided_slice_2:output:0/gru_9/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/BiasAddBiasAdd"gru_9/gru_cell_27/MatMul:product:0"gru_9/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!gru_9/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
gru_9/gru_cell_27/splitSplit*gru_9/gru_cell_27/split/split_dim:output:0"gru_9/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)gru_9/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp2gru_9_gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_9/gru_cell_27/MatMul_1MatMulgru_9/zeros:output:01gru_9/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
gru_9/gru_cell_27/BiasAdd_1BiasAdd$gru_9/gru_cell_27/MatMul_1:product:0"gru_9/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
gru_9/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#gru_9/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/split_1SplitV$gru_9/gru_cell_27/BiasAdd_1:output:0 gru_9/gru_cell_27/Const:output:0,gru_9/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_9/gru_cell_27/addAddV2 gru_9/gru_cell_27/split:output:0"gru_9/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
gru_9/gru_cell_27/SigmoidSigmoidgru_9/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/add_1AddV2 gru_9/gru_cell_27/split:output:1"gru_9/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gru_9/gru_cell_27/Sigmoid_1Sigmoidgru_9/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/mulMulgru_9/gru_cell_27/Sigmoid_1:y:0"gru_9/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/add_2AddV2 gru_9/gru_cell_27/split:output:2gru_9/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
gru_9/gru_cell_27/ReluRelugru_9/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/mul_1Mulgru_9/gru_cell_27/Sigmoid:y:0gru_9/zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
gru_9/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
gru_9/gru_cell_27/subSub gru_9/gru_cell_27/sub/x:output:0gru_9/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/mul_2Mulgru_9/gru_cell_27/sub:z:0$gru_9/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/gru_cell_27/add_3AddV2gru_9/gru_cell_27/mul_1:z:0gru_9/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
#gru_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ê
gru_9/TensorArrayV2_1TensorListReserve,gru_9/TensorArrayV2_1/element_shape:output:0gru_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒL

gru_9/timeConst*
_output_shapes
: *
dtype0*
value	B : i
gru_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿZ
gru_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
gru_9/whileWhile!gru_9/while/loop_counter:output:0'gru_9/while/maximum_iterations:output:0gru_9/time:output:0gru_9/TensorArrayV2_1:handle:0gru_9/zeros:output:0gru_9/strided_slice_1:output:0=gru_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0)gru_9_gru_cell_27_readvariableop_resource0gru_9_gru_cell_27_matmul_readvariableop_resource2gru_9_gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *#
bodyR
gru_9_while_body_109928*#
condR
gru_9_while_cond_109927*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
6gru_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ô
(gru_9/TensorArrayV2Stack/TensorListStackTensorListStackgru_9/while:output:3?gru_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0n
gru_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿg
gru_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: g
gru_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¥
gru_9/strided_slice_3StridedSlice1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0$gru_9/strided_slice_3/stack:output:0&gru_9/strided_slice_3/stack_1:output:0&gru_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskk
gru_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ¨
gru_9/transpose_1	Transpose1gru_9/TensorArrayV2Stack/TensorListStack:tensor:0gru_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    \
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout_9/dropout/MulMulgru_9/strided_slice_3:output:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
dropout_9/dropout/ShapeShapegru_9/strided_slice_3:output:0*
T0*
_output_shapes
: 
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>Ä
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_4/MatMulMatMuldropout_9/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp(^gru_8/gru_cell_26/MatMul/ReadVariableOp*^gru_8/gru_cell_26/MatMul_1/ReadVariableOp!^gru_8/gru_cell_26/ReadVariableOp^gru_8/while(^gru_9/gru_cell_27/MatMul/ReadVariableOp*^gru_9/gru_cell_27/MatMul_1/ReadVariableOp!^gru_9/gru_cell_27/ReadVariableOp^gru_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2R
'gru_8/gru_cell_26/MatMul/ReadVariableOp'gru_8/gru_cell_26/MatMul/ReadVariableOp2V
)gru_8/gru_cell_26/MatMul_1/ReadVariableOp)gru_8/gru_cell_26/MatMul_1/ReadVariableOp2D
 gru_8/gru_cell_26/ReadVariableOp gru_8/gru_cell_26/ReadVariableOp2
gru_8/whilegru_8/while2R
'gru_9/gru_cell_27/MatMul/ReadVariableOp'gru_9/gru_cell_27/MatMul/ReadVariableOp2V
)gru_9/gru_cell_27/MatMul_1/ReadVariableOp)gru_9/gru_cell_27/MatMul_1/ReadVariableOp2D
 gru_9/gru_cell_27/ReadVariableOp gru_9/gru_cell_27/ReadVariableOp2
gru_9/whilegru_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
=

while_body_109113
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_26_readvariableop_resource_0:	E
2while_gru_cell_26_matmul_readvariableop_resource_0:	G
4while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_26_readvariableop_resource:	C
0while_gru_cell_26_matmul_readvariableop_resource:	E
2while_gru_cell_26_matmul_1_readvariableop_resource:	2¢'while/gru_cell_26/MatMul/ReadVariableOp¢)while/gru_cell_26/MatMul_1/ReadVariableOp¢ while/gru_cell_26/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
while/gru_cell_26/ReluReluwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0$while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
÷
´
&__inference_gru_9_layer_call_fn_110781

inputs
unknown:
	unknown_0:2
	unknown_1:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_109004o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Ú
ª
while_cond_109112
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_109112___redundant_placeholder04
0while_while_cond_109112___redundant_placeholder14
0while_while_cond_109112___redundant_placeholder24
0while_while_cond_109112___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
Ú
ª
while_cond_110620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110620___redundant_placeholder04
0while_while_cond_110620___redundant_placeholder14
0while_while_cond_110620___redundant_placeholder24
0while_while_cond_110620___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
=

while_body_110998
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_27_readvariableop_resource_0:D
2while_gru_cell_27_matmul_readvariableop_resource_0:2F
4while_gru_cell_27_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_27_readvariableop_resource:B
0while_gru_cell_27_matmul_readvariableop_resource:2D
2while_gru_cell_27_matmul_1_readvariableop_resource:¢'while/gru_cell_27/MatMul/ReadVariableOp¢)while/gru_cell_27/MatMul_1/ReadVariableOp¢ while/gru_cell_27/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0·
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0 while/gru_cell_27/Const:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
while/gru_cell_27/ReluReluwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0$while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ

while/NoOpNoOp(^while/gru_cell_27/MatMul/ReadVariableOp*^while/gru_cell_27/MatMul_1/ReadVariableOp!^while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_27/MatMul/ReadVariableOp'while/gru_cell_27/MatMul/ReadVariableOp2V
)while/gru_cell_27/MatMul_1/ReadVariableOp)while/gru_cell_27/MatMul_1/ReadVariableOp2D
 while/gru_cell_27/ReadVariableOp while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
=

while_body_108665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_27_readvariableop_resource_0:D
2while_gru_cell_27_matmul_readvariableop_resource_0:2F
4while_gru_cell_27_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_27_readvariableop_resource:B
0while_gru_cell_27_matmul_readvariableop_resource:2D
2while_gru_cell_27_matmul_1_readvariableop_resource:¢'while/gru_cell_27/MatMul/ReadVariableOp¢)while/gru_cell_27/MatMul_1/ReadVariableOp¢ while/gru_cell_27/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0·
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0 while/gru_cell_27/Const:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
while/gru_cell_27/ReluReluwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0$while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ

while/NoOpNoOp(^while/gru_cell_27/MatMul/ReadVariableOp*^while/gru_cell_27/MatMul_1/ReadVariableOp!^while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_27/MatMul/ReadVariableOp'while/gru_cell_27/MatMul/ReadVariableOp2V
)while/gru_cell_27/MatMul_1/ReadVariableOp)while/gru_cell_27/MatMul_1/ReadVariableOp2D
 while/gru_cell_27/ReadVariableOp while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
7

__inference__traced_save_111743
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop7
3savev2_gru_8_gru_cell_26_kernel_read_readvariableopA
=savev2_gru_8_gru_cell_26_recurrent_kernel_read_readvariableop5
1savev2_gru_8_gru_cell_26_bias_read_readvariableop7
3savev2_gru_9_gru_cell_27_kernel_read_readvariableopA
=savev2_gru_9_gru_cell_27_recurrent_kernel_read_readvariableop5
1savev2_gru_9_gru_cell_27_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_4_bias_rms_read_readvariableopC
?savev2_rmsprop_gru_8_gru_cell_26_kernel_rms_read_readvariableopM
Isavev2_rmsprop_gru_8_gru_cell_26_recurrent_kernel_rms_read_readvariableopA
=savev2_rmsprop_gru_8_gru_cell_26_bias_rms_read_readvariableopC
?savev2_rmsprop_gru_9_gru_cell_27_kernel_rms_read_readvariableopM
Isavev2_rmsprop_gru_9_gru_cell_27_recurrent_kernel_rms_read_readvariableopA
=savev2_rmsprop_gru_9_gru_cell_27_bias_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¯

value¥
B¢
B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop3savev2_gru_8_gru_cell_26_kernel_read_readvariableop=savev2_gru_8_gru_cell_26_recurrent_kernel_read_readvariableop1savev2_gru_8_gru_cell_26_bias_read_readvariableop3savev2_gru_9_gru_cell_27_kernel_read_readvariableop=savev2_gru_9_gru_cell_27_recurrent_kernel_read_readvariableop1savev2_gru_9_gru_cell_27_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop3savev2_rmsprop_dense_4_bias_rms_read_readvariableop?savev2_rmsprop_gru_8_gru_cell_26_kernel_rms_read_readvariableopIsavev2_rmsprop_gru_8_gru_cell_26_recurrent_kernel_rms_read_readvariableop=savev2_rmsprop_gru_8_gru_cell_26_bias_rms_read_readvariableop?savev2_rmsprop_gru_9_gru_cell_27_kernel_rms_read_readvariableopIsavev2_rmsprop_gru_9_gru_cell_27_recurrent_kernel_rms_read_readvariableop=savev2_rmsprop_gru_9_gru_cell_27_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Å
_input_shapes³
°: ::: : : : : :	:	2:	:2::: : :::	:	2:	:2::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:%	!

_output_shapes
:	2:%
!

_output_shapes
:	:$ 

_output_shapes

:2:$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	2:%!

_output_shapes
:	:$ 

_output_shapes

:2:$ 

_output_shapes

::$ 

_output_shapes

::

_output_shapes
: 
=

while_body_110468
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_26_readvariableop_resource_0:	E
2while_gru_cell_26_matmul_readvariableop_resource_0:	G
4while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_26_readvariableop_resource:	C
0while_gru_cell_26_matmul_readvariableop_resource:	E
2while_gru_cell_26_matmul_1_readvariableop_resource:	2¢'while/gru_cell_26/MatMul/ReadVariableOp¢)while/gru_cell_26/MatMul_1/ReadVariableOp¢ while/gru_cell_26/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
while/gru_cell_26/ReluReluwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0$while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
²
F
*__inference_dropout_8_layer_call_fn_110715

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_108600d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
Æ	
ô
C__inference_dense_4_layer_call_and_return_conditional_losses_111439

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
=

while_body_111304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_27_readvariableop_resource_0:D
2while_gru_cell_27_matmul_readvariableop_resource_0:2F
4while_gru_cell_27_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_27_readvariableop_resource:B
0while_gru_cell_27_matmul_readvariableop_resource:2D
2while_gru_cell_27_matmul_1_readvariableop_resource:¢'while/gru_cell_27/MatMul/ReadVariableOp¢)while/gru_cell_27/MatMul_1/ReadVariableOp¢ while/gru_cell_27/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0·
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0 while/gru_cell_27/Const:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
while/gru_cell_27/ReluReluwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0$while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ

while/NoOpNoOp(^while/gru_cell_27/MatMul/ReadVariableOp*^while/gru_cell_27/MatMul_1/ReadVariableOp!^while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_27/MatMul/ReadVariableOp'while/gru_cell_27/MatMul/ReadVariableOp2V
)while/gru_cell_27/MatMul_1/ReadVariableOp)while/gru_cell_27/MatMul_1/ReadVariableOp2D
 while/gru_cell_27/ReadVariableOp while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ú
ª
while_cond_108914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108914___redundant_placeholder04
0while_while_cond_108914___redundant_placeholder14
0while_while_cond_108914___redundant_placeholder24
0while_while_cond_108914___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
M

A__inference_gru_9_layer_call_and_return_conditional_losses_109004

inputs5
#gru_cell_27_readvariableop_resource:<
*gru_cell_27_matmul_readvariableop_resource:2>
,gru_cell_27_matmul_1_readvariableop_resource:
identity¢!gru_cell_27/MatMul/ReadVariableOp¢#gru_cell_27/MatMul_1/ReadVariableOp¢gru_cell_27/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask~
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿh
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_cell_27/ReluRelugru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108915*
condR
while_cond_108914*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp"^gru_cell_27/MatMul/ReadVariableOp$^gru_cell_27/MatMul_1/ReadVariableOp^gru_cell_27/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : : 2F
!gru_cell_27/MatMul/ReadVariableOp!gru_cell_27/MatMul/ReadVariableOp2J
#gru_cell_27/MatMul_1/ReadVariableOp#gru_cell_27/MatMul_1/ReadVariableOp28
gru_cell_27/ReadVariableOpgru_cell_27/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs
ó^
°
"__inference__traced_restore_111822
file_prefix1
assignvariableop_dense_4_kernel:-
assignvariableop_1_dense_4_bias:)
assignvariableop_2_rmsprop_iter:	 *
 assignvariableop_3_rmsprop_decay: 2
(assignvariableop_4_rmsprop_learning_rate: -
#assignvariableop_5_rmsprop_momentum: (
assignvariableop_6_rmsprop_rho: >
+assignvariableop_7_gru_8_gru_cell_26_kernel:	H
5assignvariableop_8_gru_8_gru_cell_26_recurrent_kernel:	2<
)assignvariableop_9_gru_8_gru_cell_26_bias:	>
,assignvariableop_10_gru_9_gru_cell_27_kernel:2H
6assignvariableop_11_gru_9_gru_cell_27_recurrent_kernel:<
*assignvariableop_12_gru_9_gru_cell_27_bias:#
assignvariableop_13_total: #
assignvariableop_14_count: @
.assignvariableop_15_rmsprop_dense_4_kernel_rms::
,assignvariableop_16_rmsprop_dense_4_bias_rms:K
8assignvariableop_17_rmsprop_gru_8_gru_cell_26_kernel_rms:	U
Bassignvariableop_18_rmsprop_gru_8_gru_cell_26_recurrent_kernel_rms:	2I
6assignvariableop_19_rmsprop_gru_8_gru_cell_26_bias_rms:	J
8assignvariableop_20_rmsprop_gru_9_gru_cell_27_kernel_rms:2T
Bassignvariableop_21_rmsprop_gru_9_gru_cell_27_recurrent_kernel_rms:H
6assignvariableop_22_rmsprop_gru_9_gru_cell_27_bias_rms:
identity_24¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¯

value¥
B¢
B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_rmsprop_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_rmsprop_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rmsprop_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_rmsprop_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_rhoIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_gru_8_gru_cell_26_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_8AssignVariableOp5assignvariableop_8_gru_8_gru_cell_26_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp)assignvariableop_9_gru_8_gru_cell_26_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp,assignvariableop_10_gru_9_gru_cell_27_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_11AssignVariableOp6assignvariableop_11_gru_9_gru_cell_27_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp*assignvariableop_12_gru_9_gru_cell_27_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp.assignvariableop_15_rmsprop_dense_4_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp,assignvariableop_16_rmsprop_dense_4_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_17AssignVariableOp8assignvariableop_17_rmsprop_gru_8_gru_cell_26_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_18AssignVariableOpBassignvariableop_18_rmsprop_gru_8_gru_cell_26_recurrent_kernel_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_19AssignVariableOp6assignvariableop_19_rmsprop_gru_8_gru_cell_26_bias_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_20AssignVariableOp8assignvariableop_20_rmsprop_gru_9_gru_cell_27_kernel_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:³
AssignVariableOp_21AssignVariableOpBassignvariableop_21_rmsprop_gru_9_gru_cell_27_recurrent_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_22AssignVariableOp6assignvariableop_22_rmsprop_gru_9_gru_cell_27_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 É
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_24IdentityIdentity_23:output:0^NoOp_1*
T0*
_output_shapes
: ¶
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_24Identity_24:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ÁM

A__inference_gru_9_layer_call_and_return_conditional_losses_110934
inputs_05
#gru_cell_27_readvariableop_resource:<
*gru_cell_27_matmul_readvariableop_resource:2>
,gru_cell_27_matmul_1_readvariableop_resource:
identity¢!gru_cell_27/MatMul/ReadVariableOp¢#gru_cell_27/MatMul_1/ReadVariableOp¢gru_cell_27/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask~
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿh
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_cell_27/ReluRelugru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110845*
condR
while_cond_110844*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp"^gru_cell_27/MatMul/ReadVariableOp$^gru_cell_27/MatMul_1/ReadVariableOp^gru_cell_27/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2F
!gru_cell_27/MatMul/ReadVariableOp!gru_cell_27/MatMul/ReadVariableOp2J
#gru_cell_27/MatMul_1/ReadVariableOp#gru_cell_27/MatMul_1/ReadVariableOp28
gru_cell_27/ReadVariableOpgru_cell_27/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs/0
÷
Ì
H__inference_sequential_4_layer_call_and_return_conditional_losses_108786

inputs
gru_8_108588:	
gru_8_108590:	
gru_8_108592:	2
gru_9_108755:
gru_9_108757:2
gru_9_108759: 
dense_4_108780:
dense_4_108782:
identity¢dense_4/StatefulPartitionedCall¢gru_8/StatefulPartitionedCall¢gru_9/StatefulPartitionedCallû
gru_8/StatefulPartitionedCallStatefulPartitionedCallinputsgru_8_108588gru_8_108590gru_8_108592*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_108587á
dropout_8/PartitionedCallPartitionedCall&gru_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_108600
gru_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0gru_9_108755gru_9_108757gru_9_108759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_108754Ý
dropout_9/PartitionedCallPartitionedCall&gru_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_108767
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_4_108780dense_4_108782*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_108779w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp ^dense_4/StatefulPartitionedCall^gru_8/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
&__inference_gru_9_layer_call_fn_110759
inputs_0
unknown:
	unknown_0:2
	unknown_1:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_108419o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs/0
ï	
Ð
-__inference_sequential_4_layer_call_fn_109299
gru_8_input
unknown:	
	unknown_0:	
	unknown_1:	2
	unknown_2:
	unknown_3:2
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallgru_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_109259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namegru_8_input
Ø
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_108767

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
ª
while_cond_108172
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108172___redundant_placeholder04
0while_while_cond_108172___redundant_placeholder14
0while_while_cond_108172___redundant_placeholder24
0while_while_cond_108172___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ú
ª
while_cond_111150
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_111150___redundant_placeholder04
0while_while_cond_111150___redundant_placeholder14
0while_while_cond_111150___redundant_placeholder24
0while_while_cond_111150___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

Ø
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_111612

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:22
 matmul_1_readvariableop_resource:
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
U
Â
$sequential_4_gru_8_while_body_107506B
>sequential_4_gru_8_while_sequential_4_gru_8_while_loop_counterH
Dsequential_4_gru_8_while_sequential_4_gru_8_while_maximum_iterations(
$sequential_4_gru_8_while_placeholder*
&sequential_4_gru_8_while_placeholder_1*
&sequential_4_gru_8_while_placeholder_2A
=sequential_4_gru_8_while_sequential_4_gru_8_strided_slice_1_0}
ysequential_4_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_8_tensorarrayunstack_tensorlistfromtensor_0Q
>sequential_4_gru_8_while_gru_cell_26_readvariableop_resource_0:	X
Esequential_4_gru_8_while_gru_cell_26_matmul_readvariableop_resource_0:	Z
Gsequential_4_gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0:	2%
!sequential_4_gru_8_while_identity'
#sequential_4_gru_8_while_identity_1'
#sequential_4_gru_8_while_identity_2'
#sequential_4_gru_8_while_identity_3'
#sequential_4_gru_8_while_identity_4?
;sequential_4_gru_8_while_sequential_4_gru_8_strided_slice_1{
wsequential_4_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_8_tensorarrayunstack_tensorlistfromtensorO
<sequential_4_gru_8_while_gru_cell_26_readvariableop_resource:	V
Csequential_4_gru_8_while_gru_cell_26_matmul_readvariableop_resource:	X
Esequential_4_gru_8_while_gru_cell_26_matmul_1_readvariableop_resource:	2¢:sequential_4/gru_8/while/gru_cell_26/MatMul/ReadVariableOp¢<sequential_4/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp¢3sequential_4/gru_8/while/gru_cell_26/ReadVariableOp
Jsequential_4/gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
<sequential_4/gru_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemysequential_4_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_8_tensorarrayunstack_tensorlistfromtensor_0$sequential_4_gru_8_while_placeholderSsequential_4/gru_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
3sequential_4/gru_8/while/gru_cell_26/ReadVariableOpReadVariableOp>sequential_4_gru_8_while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0«
,sequential_4/gru_8/while/gru_cell_26/unstackUnpack;sequential_4/gru_8/while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numÁ
:sequential_4/gru_8/while/gru_cell_26/MatMul/ReadVariableOpReadVariableOpEsequential_4_gru_8_while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ñ
+sequential_4/gru_8/while/gru_cell_26/MatMulMatMulCsequential_4/gru_8/while/TensorArrayV2Read/TensorListGetItem:item:0Bsequential_4/gru_8/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
,sequential_4/gru_8/while/gru_cell_26/BiasAddBiasAdd5sequential_4/gru_8/while/gru_cell_26/MatMul:product:05sequential_4/gru_8/while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4sequential_4/gru_8/while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
*sequential_4/gru_8/while/gru_cell_26/splitSplit=sequential_4/gru_8/while/gru_cell_26/split/split_dim:output:05sequential_4/gru_8/while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitÅ
<sequential_4/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOpGsequential_4_gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0Ø
-sequential_4/gru_8/while/gru_cell_26/MatMul_1MatMul&sequential_4_gru_8_while_placeholder_2Dsequential_4/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
.sequential_4/gru_8/while/gru_cell_26/BiasAdd_1BiasAdd7sequential_4/gru_8/while/gru_cell_26/MatMul_1:product:05sequential_4/gru_8/while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_4/gru_8/while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ
6sequential_4/gru_8/while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
,sequential_4/gru_8/while/gru_cell_26/split_1SplitV7sequential_4/gru_8/while/gru_cell_26/BiasAdd_1:output:03sequential_4/gru_8/while/gru_cell_26/Const:output:0?sequential_4/gru_8/while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splitÏ
(sequential_4/gru_8/while/gru_cell_26/addAddV23sequential_4/gru_8/while/gru_cell_26/split:output:05sequential_4/gru_8/while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
,sequential_4/gru_8/while/gru_cell_26/SigmoidSigmoid,sequential_4/gru_8/while/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ñ
*sequential_4/gru_8/while/gru_cell_26/add_1AddV23sequential_4/gru_8/while/gru_cell_26/split:output:15sequential_4/gru_8/while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
.sequential_4/gru_8/while/gru_cell_26/Sigmoid_1Sigmoid.sequential_4/gru_8/while/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ì
(sequential_4/gru_8/while/gru_cell_26/mulMul2sequential_4/gru_8/while/gru_cell_26/Sigmoid_1:y:05sequential_4/gru_8/while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2È
*sequential_4/gru_8/while/gru_cell_26/add_2AddV23sequential_4/gru_8/while/gru_cell_26/split:output:2,sequential_4/gru_8/while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
)sequential_4/gru_8/while/gru_cell_26/ReluRelu.sequential_4/gru_8/while/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2½
*sequential_4/gru_8/while/gru_cell_26/mul_1Mul0sequential_4/gru_8/while/gru_cell_26/Sigmoid:y:0&sequential_4_gru_8_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2o
*sequential_4/gru_8/while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
(sequential_4/gru_8/while/gru_cell_26/subSub3sequential_4/gru_8/while/gru_cell_26/sub/x:output:00sequential_4/gru_8/while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ê
*sequential_4/gru_8/while/gru_cell_26/mul_2Mul,sequential_4/gru_8/while/gru_cell_26/sub:z:07sequential_4/gru_8/while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å
*sequential_4/gru_8/while/gru_cell_26/add_3AddV2.sequential_4/gru_8/while/gru_cell_26/mul_1:z:0.sequential_4/gru_8/while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
=sequential_4/gru_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&sequential_4_gru_8_while_placeholder_1$sequential_4_gru_8_while_placeholder.sequential_4/gru_8/while/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒ`
sequential_4/gru_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_4/gru_8/while/addAddV2$sequential_4_gru_8_while_placeholder'sequential_4/gru_8/while/add/y:output:0*
T0*
_output_shapes
: b
 sequential_4/gru_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :³
sequential_4/gru_8/while/add_1AddV2>sequential_4_gru_8_while_sequential_4_gru_8_while_loop_counter)sequential_4/gru_8/while/add_1/y:output:0*
T0*
_output_shapes
: 
!sequential_4/gru_8/while/IdentityIdentity"sequential_4/gru_8/while/add_1:z:0^sequential_4/gru_8/while/NoOp*
T0*
_output_shapes
: ¶
#sequential_4/gru_8/while/Identity_1IdentityDsequential_4_gru_8_while_sequential_4_gru_8_while_maximum_iterations^sequential_4/gru_8/while/NoOp*
T0*
_output_shapes
: 
#sequential_4/gru_8/while/Identity_2Identity sequential_4/gru_8/while/add:z:0^sequential_4/gru_8/while/NoOp*
T0*
_output_shapes
: Ò
#sequential_4/gru_8/while/Identity_3IdentityMsequential_4/gru_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/gru_8/while/NoOp*
T0*
_output_shapes
: :éèÒ±
#sequential_4/gru_8/while/Identity_4Identity.sequential_4/gru_8/while/gru_cell_26/add_3:z:0^sequential_4/gru_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_4/gru_8/while/NoOpNoOp;^sequential_4/gru_8/while/gru_cell_26/MatMul/ReadVariableOp=^sequential_4/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp4^sequential_4/gru_8/while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
Esequential_4_gru_8_while_gru_cell_26_matmul_1_readvariableop_resourceGsequential_4_gru_8_while_gru_cell_26_matmul_1_readvariableop_resource_0"
Csequential_4_gru_8_while_gru_cell_26_matmul_readvariableop_resourceEsequential_4_gru_8_while_gru_cell_26_matmul_readvariableop_resource_0"~
<sequential_4_gru_8_while_gru_cell_26_readvariableop_resource>sequential_4_gru_8_while_gru_cell_26_readvariableop_resource_0"O
!sequential_4_gru_8_while_identity*sequential_4/gru_8/while/Identity:output:0"S
#sequential_4_gru_8_while_identity_1,sequential_4/gru_8/while/Identity_1:output:0"S
#sequential_4_gru_8_while_identity_2,sequential_4/gru_8/while/Identity_2:output:0"S
#sequential_4_gru_8_while_identity_3,sequential_4/gru_8/while/Identity_3:output:0"S
#sequential_4_gru_8_while_identity_4,sequential_4/gru_8/while/Identity_4:output:0"|
;sequential_4_gru_8_while_sequential_4_gru_8_strided_slice_1=sequential_4_gru_8_while_sequential_4_gru_8_strided_slice_1_0"ô
wsequential_4_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_8_tensorarrayunstack_tensorlistfromtensorysequential_4_gru_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_gru_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2x
:sequential_4/gru_8/while/gru_cell_26/MatMul/ReadVariableOp:sequential_4/gru_8/while/gru_cell_26/MatMul/ReadVariableOp2|
<sequential_4/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp<sequential_4/gru_8/while/gru_cell_26/MatMul_1/ReadVariableOp2j
3sequential_4/gru_8/while/gru_cell_26/ReadVariableOp3sequential_4/gru_8/while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ú
ª
while_cond_110844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110844___redundant_placeholder04
0while_while_cond_110844___redundant_placeholder14
0while_while_cond_110844___redundant_placeholder24
0while_while_cond_110844___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

Ñ
H__inference_sequential_4_layer_call_and_return_conditional_losses_109324
gru_8_input
gru_8_109302:	
gru_8_109304:	
gru_8_109306:	2
gru_9_109310:
gru_9_109312:2
gru_9_109314: 
dense_4_109318:
dense_4_109320:
identity¢dense_4/StatefulPartitionedCall¢gru_8/StatefulPartitionedCall¢gru_9/StatefulPartitionedCall
gru_8/StatefulPartitionedCallStatefulPartitionedCallgru_8_inputgru_8_109302gru_8_109304gru_8_109306*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_108587á
dropout_8/PartitionedCallPartitionedCall&gru_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_108600
gru_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0gru_9_109310gru_9_109312gru_9_109314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_108754Ý
dropout_9/PartitionedCallPartitionedCall&gru_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_108767
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_4_109318dense_4_109320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_108779w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
NoOpNoOp ^dense_4/StatefulPartitionedCall^gru_8/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namegru_8_input
 
¯
while_body_108173
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
while_gru_cell_27_108195_0:,
while_gru_cell_27_108197_0:2,
while_gru_cell_27_108199_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
while_gru_cell_27_108195:*
while_gru_cell_27_108197:2*
while_gru_cell_27_108199:¢)while/gru_cell_27/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
)while/gru_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_27_108195_0while_gru_cell_27_108197_0while_gru_cell_27_108199_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108160Û
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/gru_cell_27/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity2while/gru_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx

while/NoOpNoOp*^while/gru_cell_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "6
while_gru_cell_27_108195while_gru_cell_27_108195_0"6
while_gru_cell_27_108197while_gru_cell_27_108197_0"6
while_gru_cell_27_108199while_gru_cell_27_108199_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/gru_cell_27/StatefulPartitionedCall)while/gru_cell_27/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÑD
¶	
gru_9_while_body_109611(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2'
#gru_9_while_gru_9_strided_slice_1_0c
_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0C
1gru_9_while_gru_cell_27_readvariableop_resource_0:J
8gru_9_while_gru_cell_27_matmul_readvariableop_resource_0:2L
:gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0:
gru_9_while_identity
gru_9_while_identity_1
gru_9_while_identity_2
gru_9_while_identity_3
gru_9_while_identity_4%
!gru_9_while_gru_9_strided_slice_1a
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensorA
/gru_9_while_gru_cell_27_readvariableop_resource:H
6gru_9_while_gru_cell_27_matmul_readvariableop_resource:2J
8gru_9_while_gru_cell_27_matmul_1_readvariableop_resource:¢-gru_9/while/gru_cell_27/MatMul/ReadVariableOp¢/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp¢&gru_9/while/gru_cell_27/ReadVariableOp
=gru_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Ä
/gru_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0gru_9_while_placeholderFgru_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
&gru_9/while/gru_cell_27/ReadVariableOpReadVariableOp1gru_9_while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
gru_9/while/gru_cell_27/unstackUnpack.gru_9/while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num¦
-gru_9/while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp8gru_9_while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0É
gru_9/while/gru_cell_27/MatMulMatMul6gru_9/while/TensorArrayV2Read/TensorListGetItem:item:05gru_9/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
gru_9/while/gru_cell_27/BiasAddBiasAdd(gru_9/while/gru_cell_27/MatMul:product:0(gru_9/while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
'gru_9/while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿë
gru_9/while/gru_cell_27/splitSplit0gru_9/while/gru_cell_27/split/split_dim:output:0(gru_9/while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitª
/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp:gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0°
 gru_9/while/gru_cell_27/MatMul_1MatMulgru_9_while_placeholder_27gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ´
!gru_9/while/gru_cell_27/BiasAdd_1BiasAdd*gru_9/while/gru_cell_27/MatMul_1:product:0(gru_9/while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
gru_9/while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿt
)gru_9/while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¦
gru_9/while/gru_cell_27/split_1SplitV*gru_9/while/gru_cell_27/BiasAdd_1:output:0&gru_9/while/gru_cell_27/Const:output:02gru_9/while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split¨
gru_9/while/gru_cell_27/addAddV2&gru_9/while/gru_cell_27/split:output:0(gru_9/while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_9/while/gru_cell_27/SigmoidSigmoidgru_9/while/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
gru_9/while/gru_cell_27/add_1AddV2&gru_9/while/gru_cell_27/split:output:1(gru_9/while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!gru_9/while/gru_cell_27/Sigmoid_1Sigmoid!gru_9/while/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
gru_9/while/gru_cell_27/mulMul%gru_9/while/gru_cell_27/Sigmoid_1:y:0(gru_9/while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
gru_9/while/gru_cell_27/add_2AddV2&gru_9/while/gru_cell_27/split:output:2gru_9/while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
gru_9/while/gru_cell_27/ReluRelu!gru_9/while/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/while/gru_cell_27/mul_1Mul#gru_9/while/gru_cell_27/Sigmoid:y:0gru_9_while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
gru_9/while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¡
gru_9/while/gru_cell_27/subSub&gru_9/while/gru_cell_27/sub/x:output:0#gru_9/while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
gru_9/while/gru_cell_27/mul_2Mulgru_9/while/gru_cell_27/sub:z:0*gru_9/while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_9/while/gru_cell_27/add_3AddV2!gru_9/while/gru_cell_27/mul_1:z:0!gru_9/while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
0gru_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_9_while_placeholder_1gru_9_while_placeholder!gru_9/while/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒS
gru_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :n
gru_9/while/addAddV2gru_9_while_placeholdergru_9/while/add/y:output:0*
T0*
_output_shapes
: U
gru_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
gru_9/while/add_1AddV2$gru_9_while_gru_9_while_loop_countergru_9/while/add_1/y:output:0*
T0*
_output_shapes
: k
gru_9/while/IdentityIdentitygru_9/while/add_1:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: 
gru_9/while/Identity_1Identity*gru_9_while_gru_9_while_maximum_iterations^gru_9/while/NoOp*
T0*
_output_shapes
: k
gru_9/while/Identity_2Identitygru_9/while/add:z:0^gru_9/while/NoOp*
T0*
_output_shapes
: «
gru_9/while/Identity_3Identity@gru_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^gru_9/while/NoOp*
T0*
_output_shapes
: :éèÒ
gru_9/while/Identity_4Identity!gru_9/while/gru_cell_27/add_3:z:0^gru_9/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÝ
gru_9/while/NoOpNoOp.^gru_9/while/gru_cell_27/MatMul/ReadVariableOp0^gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp'^gru_9/while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "H
!gru_9_while_gru_9_strided_slice_1#gru_9_while_gru_9_strided_slice_1_0"v
8gru_9_while_gru_cell_27_matmul_1_readvariableop_resource:gru_9_while_gru_cell_27_matmul_1_readvariableop_resource_0"r
6gru_9_while_gru_cell_27_matmul_readvariableop_resource8gru_9_while_gru_cell_27_matmul_readvariableop_resource_0"d
/gru_9_while_gru_cell_27_readvariableop_resource1gru_9_while_gru_cell_27_readvariableop_resource_0"5
gru_9_while_identitygru_9/while/Identity:output:0"9
gru_9_while_identity_1gru_9/while/Identity_1:output:0"9
gru_9_while_identity_2gru_9/while/Identity_2:output:0"9
gru_9_while_identity_3gru_9/while/Identity_3:output:0"9
gru_9_while_identity_4gru_9/while/Identity_4:output:0"À
]gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_gru_9_while_tensorarrayv2read_tensorlistgetitem_gru_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2^
-gru_9/while/gru_cell_27/MatMul/ReadVariableOp-gru_9/while/gru_cell_27/MatMul/ReadVariableOp2b
/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp/gru_9/while/gru_cell_27/MatMul_1/ReadVariableOp2P
&gru_9/while/gru_cell_27/ReadVariableOp&gru_9/while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
þ
Ö
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_108160

inputs

states)
readvariableop_resource:0
matmul_readvariableop_resource:22
 matmul_1_readvariableop_resource:
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates

Û
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_111506

inputs
states_0*
readvariableop_resource:	1
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0
ï	
Ð
-__inference_sequential_4_layer_call_fn_108805
gru_8_input
unknown:	
	unknown_0:	
	unknown_1:	2
	unknown_2:
	unknown_3:2
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallgru_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_108786o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_namegru_8_input
Ø
c
E__inference_dropout_9_layer_call_and_return_conditional_losses_111408

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·

Û
,__inference_gru_cell_26_layer_call_fn_111453

inputs
states_0
unknown:	
	unknown_0:	
	unknown_1:	2
identity

identity_1¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107822o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
states/0
=

while_body_110621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0>
+while_gru_cell_26_readvariableop_resource_0:	E
2while_gru_cell_26_matmul_readvariableop_resource_0:	G
4while_gru_cell_26_matmul_1_readvariableop_resource_0:	2
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor<
)while_gru_cell_26_readvariableop_resource:	C
0while_gru_cell_26_matmul_readvariableop_resource:	E
2while_gru_cell_26_matmul_1_readvariableop_resource:	2¢'while/gru_cell_26/MatMul/ReadVariableOp¢)while/gru_cell_26/MatMul_1/ReadVariableOp¢ while/gru_cell_26/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
 while/gru_cell_26/ReadVariableOpReadVariableOp+while_gru_cell_26_readvariableop_resource_0*
_output_shapes
:	*
dtype0
while/gru_cell_26/unstackUnpack(while/gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
'while/gru_cell_26/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/gru_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/BiasAddBiasAdd"while/gru_cell_26/MatMul:product:0"while/gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_26/splitSplit*while/gru_cell_26/split/split_dim:output:0"while/gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
)while/gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_26_matmul_1_readvariableop_resource_0*
_output_shapes
:	2*
dtype0
while/gru_cell_26/MatMul_1MatMulwhile_placeholder_21while/gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
while/gru_cell_26/BiasAdd_1BiasAdd$while/gru_cell_26/MatMul_1:product:0"while/gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿn
#while/gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_26/split_1SplitV$while/gru_cell_26/BiasAdd_1:output:0 while/gru_cell_26/Const:output:0,while/gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
while/gru_cell_26/addAddV2 while/gru_cell_26/split:output:0"while/gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
while/gru_cell_26/SigmoidSigmoidwhile/gru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_1AddV2 while/gru_cell_26/split:output:1"while/gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2u
while/gru_cell_26/Sigmoid_1Sigmoidwhile/gru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mulMulwhile/gru_cell_26/Sigmoid_1:y:0"while/gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_2AddV2 while/gru_cell_26/split:output:2while/gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
while/gru_cell_26/ReluReluwhile/gru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_1Mulwhile/gru_cell_26/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
while/gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_26/subSub while/gru_cell_26/sub/x:output:0while/gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/mul_2Mulwhile/gru_cell_26/sub:z:0$while/gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/gru_cell_26/add_3AddV2while/gru_cell_26/mul_1:z:0while/gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Ä
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_26/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_26/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Å

while/NoOpNoOp(^while/gru_cell_26/MatMul/ReadVariableOp*^while/gru_cell_26/MatMul_1/ReadVariableOp!^while/gru_cell_26/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_26_matmul_1_readvariableop_resource4while_gru_cell_26_matmul_1_readvariableop_resource_0"f
0while_gru_cell_26_matmul_readvariableop_resource2while_gru_cell_26_matmul_readvariableop_resource_0"X
)while_gru_cell_26_readvariableop_resource+while_gru_cell_26_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : 2R
'while/gru_cell_26/MatMul/ReadVariableOp'while/gru_cell_26/MatMul/ReadVariableOp2V
)while/gru_cell_26/MatMul_1/ReadVariableOp)while/gru_cell_26/MatMul_1/ReadVariableOp2D
 while/gru_cell_26/ReadVariableOp while/gru_cell_26/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
: 
Ú
ª
while_cond_108354
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108354___redundant_placeholder04
0while_while_cond_108354___redundant_placeholder14
0while_while_cond_108354___redundant_placeholder24
0while_while_cond_108354___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ú
ª
while_cond_110467
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_110467___redundant_placeholder04
0while_while_cond_110467___redundant_placeholder14
0while_while_cond_110467___redundant_placeholder24
0while_while_cond_110467___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
ÁM

A__inference_gru_9_layer_call_and_return_conditional_losses_111087
inputs_05
#gru_cell_27_readvariableop_resource:<
*gru_cell_27_matmul_readvariableop_resource:2>
,gru_cell_27_matmul_1_readvariableop_resource:
identity¢!gru_cell_27/MatMul/ReadVariableOp¢#gru_cell_27/MatMul_1/ReadVariableOp¢gru_cell_27/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask~
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿh
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_cell_27/ReluRelugru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110998*
condR
while_cond_110997*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp"^gru_cell_27/MatMul/ReadVariableOp$^gru_cell_27/MatMul_1/ReadVariableOp^gru_cell_27/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 2F
!gru_cell_27/MatMul/ReadVariableOp!gru_cell_27/MatMul/ReadVariableOp2J
#gru_cell_27/MatMul_1/ReadVariableOp#gru_cell_27/MatMul_1/ReadVariableOp28
gru_cell_27/ReadVariableOpgru_cell_27/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs/0
ë

H__inference_sequential_4_layer_call_and_return_conditional_losses_109259

inputs
gru_8_109237:	
gru_8_109239:	
gru_8_109241:	2
gru_9_109245:
gru_9_109247:2
gru_9_109249: 
dense_4_109253:
dense_4_109255:
identity¢dense_4/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢gru_8/StatefulPartitionedCall¢gru_9/StatefulPartitionedCallû
gru_8/StatefulPartitionedCallStatefulPartitionedCallinputsgru_8_109237gru_8_109239gru_8_109241*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_109202ñ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall&gru_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_109033
gru_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0gru_9_109245gru_9_109247gru_9_109249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_109004
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall&gru_9/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_9_layer_call_and_return_conditional_losses_108835
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_4_109253dense_4_109255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_108779w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^gru_8/StatefulPartitionedCall^gru_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2>
gru_8/StatefulPartitionedCallgru_8/StatefulPartitionedCall2>
gru_9/StatefulPartitionedCallgru_9/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
M

A__inference_gru_8_layer_call_and_return_conditional_losses_110710

inputs6
#gru_cell_26_readvariableop_resource:	=
*gru_cell_26_matmul_readvariableop_resource:	?
,gru_cell_26_matmul_1_readvariableop_resource:	2
identity¢!gru_cell_26/MatMul/ReadVariableOp¢#gru_cell_26/MatMul_1/ReadVariableOp¢gru_cell_26/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿh
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_cell_26/ReluRelugru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_110621*
condR
while_cond_110620*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2µ
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
=

while_body_110845
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0=
+while_gru_cell_27_readvariableop_resource_0:D
2while_gru_cell_27_matmul_readvariableop_resource_0:2F
4while_gru_cell_27_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor;
)while_gru_cell_27_readvariableop_resource:B
0while_gru_cell_27_matmul_readvariableop_resource:2D
2while_gru_cell_27_matmul_1_readvariableop_resource:¢'while/gru_cell_27/MatMul/ReadVariableOp¢)while/gru_cell_27/MatMul_1/ReadVariableOp¢ while/gru_cell_27/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0
 while/gru_cell_27/ReadVariableOpReadVariableOp+while_gru_cell_27_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/unstackUnpack(while/gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
'while/gru_cell_27/MatMul/ReadVariableOpReadVariableOp2while_gru_cell_27_matmul_readvariableop_resource_0*
_output_shapes

:2*
dtype0·
while/gru_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/BiasAddBiasAdd"while/gru_cell_27/MatMul:product:0"while/gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
!while/gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÙ
while/gru_cell_27/splitSplit*while/gru_cell_27/split/split_dim:output:0"while/gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
)while/gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp4while_gru_cell_27_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0
while/gru_cell_27/MatMul_1MatMulwhile_placeholder_21while/gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
while/gru_cell_27/BiasAdd_1BiasAdd$while/gru_cell_27/MatMul_1:product:0"while/gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
while/gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿn
#while/gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/split_1SplitV$while/gru_cell_27/BiasAdd_1:output:0 while/gru_cell_27/Const:output:0,while/gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
while/gru_cell_27/addAddV2 while/gru_cell_27/split:output:0"while/gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
while/gru_cell_27/SigmoidSigmoidwhile/gru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_1AddV2 while/gru_cell_27/split:output:1"while/gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
while/gru_cell_27/Sigmoid_1Sigmoidwhile/gru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mulMulwhile/gru_cell_27/Sigmoid_1:y:0"while/gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_2AddV2 while/gru_cell_27/split:output:2while/gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
while/gru_cell_27/ReluReluwhile/gru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_1Mulwhile/gru_cell_27/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
while/gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
while/gru_cell_27/subSub while/gru_cell_27/sub/x:output:0while/gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/mul_2Mulwhile/gru_cell_27/sub:z:0$while/gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/gru_cell_27/add_3AddV2while/gru_cell_27/mul_1:z:0while/gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÄ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell_27/add_3:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒx
while/Identity_4Identitywhile/gru_cell_27/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ

while/NoOpNoOp(^while/gru_cell_27/MatMul/ReadVariableOp*^while/gru_cell_27/MatMul_1/ReadVariableOp!^while/gru_cell_27/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "j
2while_gru_cell_27_matmul_1_readvariableop_resource4while_gru_cell_27_matmul_1_readvariableop_resource_0"f
0while_gru_cell_27_matmul_readvariableop_resource2while_gru_cell_27_matmul_readvariableop_resource_0"X
)while_gru_cell_27_readvariableop_resource+while_gru_cell_27_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : 2R
'while/gru_cell_27/MatMul/ReadVariableOp'while/gru_cell_27/MatMul/ReadVariableOp2V
)while/gru_cell_27/MatMul_1/ReadVariableOp)while/gru_cell_27/MatMul_1/ReadVariableOp2D
 while/gru_cell_27/ReadVariableOp while/gru_cell_27/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

¶
&__inference_gru_9_layer_call_fn_110748
inputs_0
unknown:
	unknown_0:2
	unknown_1:
identity¢StatefulPartitionedCallè
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_9_layer_call_and_return_conditional_losses_108237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
"
_user_specified_name
inputs/0
ó	
d
E__inference_dropout_9_layer_call_and_return_conditional_losses_111420

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ù
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_107965

inputs

states*
readvariableop_resource:	1
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	2
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpg
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	*
dtype0a
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
::*	
numu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_splity
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2b
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Y
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2S
mul_1MulSigmoid:y:0states*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Z

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ2: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_namestates
à	
Ë
-__inference_sequential_4_layer_call_fn_109397

inputs
unknown:	
	unknown_0:	
	unknown_1:	2
	unknown_2:
	unknown_3:2
	unknown_4:
	unknown_5:
	unknown_6:
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_109259o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü

gru_9_while_cond_109610(
$gru_9_while_gru_9_while_loop_counter.
*gru_9_while_gru_9_while_maximum_iterations
gru_9_while_placeholder
gru_9_while_placeholder_1
gru_9_while_placeholder_2*
&gru_9_while_less_gru_9_strided_slice_1@
<gru_9_while_gru_9_while_cond_109610___redundant_placeholder0@
<gru_9_while_gru_9_while_cond_109610___redundant_placeholder1@
<gru_9_while_gru_9_while_cond_109610___redundant_placeholder2@
<gru_9_while_gru_9_while_cond_109610___redundant_placeholder3
gru_9_while_identity
z
gru_9/while/LessLessgru_9_while_placeholder&gru_9_while_less_gru_9_strided_slice_1*
T0*
_output_shapes
: W
gru_9/while/IdentityIdentitygru_9/while/Less:z:0*
T0
*
_output_shapes
: "5
gru_9_while_identitygru_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¬
¹
&__inference_gru_8_layer_call_fn_110076
inputs_0
unknown:	
	unknown_0:	
	unknown_1:	2
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_gru_8_layer_call_and_return_conditional_losses_108081|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
M

A__inference_gru_8_layer_call_and_return_conditional_losses_108587

inputs6
#gru_cell_26_readvariableop_resource:	=
*gru_cell_26_matmul_readvariableop_resource:	?
,gru_cell_26_matmul_1_readvariableop_resource:	2
identity¢!gru_cell_26/MatMul/ReadVariableOp¢#gru_cell_26/MatMul_1/ReadVariableOp¢gru_cell_26/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
gru_cell_26/ReadVariableOpReadVariableOp#gru_cell_26_readvariableop_resource*
_output_shapes
:	*
dtype0y
gru_cell_26/unstackUnpack"gru_cell_26/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num
!gru_cell_26/MatMul/ReadVariableOpReadVariableOp*gru_cell_26_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gru_cell_26/MatMulMatMulstrided_slice_2:output:0)gru_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAddBiasAddgru_cell_26/MatMul:product:0gru_cell_26/unstack:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_26/splitSplit$gru_cell_26/split/split_dim:output:0gru_cell_26/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
#gru_cell_26/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_26_matmul_1_readvariableop_resource*
_output_shapes
:	2*
dtype0
gru_cell_26/MatMul_1MatMulzeros:output:0+gru_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_26/BiasAdd_1BiasAddgru_cell_26/MatMul_1:product:0gru_cell_26/unstack:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_26/ConstConst*
_output_shapes
:*
dtype0*!
valueB"2   2   ÿÿÿÿh
gru_cell_26/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_26/split_1SplitVgru_cell_26/BiasAdd_1:output:0gru_cell_26/Const:output:0&gru_cell_26/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ2*
	num_split
gru_cell_26/addAddV2gru_cell_26/split:output:0gru_cell_26/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2e
gru_cell_26/SigmoidSigmoidgru_cell_26/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/add_1AddV2gru_cell_26/split:output:1gru_cell_26/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2i
gru_cell_26/Sigmoid_1Sigmoidgru_cell_26/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mulMulgru_cell_26/Sigmoid_1:y:0gru_cell_26/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2}
gru_cell_26/add_2AddV2gru_cell_26/split:output:2gru_cell_26/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2a
gru_cell_26/ReluRelugru_cell_26/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
gru_cell_26/mul_1Mulgru_cell_26/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2V
gru_cell_26/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_26/subSubgru_cell_26/sub/x:output:0gru_cell_26/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
gru_cell_26/mul_2Mulgru_cell_26/sub:z:0gru_cell_26/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2z
gru_cell_26/add_3AddV2gru_cell_26/mul_1:z:0gru_cell_26/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_26_readvariableop_resource*gru_cell_26_matmul_readvariableop_resource,gru_cell_26_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_108498*
condR
while_cond_108497*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ2: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    b
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2µ
NoOpNoOp"^gru_cell_26/MatMul/ReadVariableOp$^gru_cell_26/MatMul_1/ReadVariableOp^gru_cell_26/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2F
!gru_cell_26/MatMul/ReadVariableOp!gru_cell_26/MatMul/ReadVariableOp2J
#gru_cell_26/MatMul_1/ReadVariableOp#gru_cell_26/MatMul_1/ReadVariableOp28
gru_cell_26/ReadVariableOpgru_cell_26/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ø
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_111651

inputs
states_0)
readvariableop_resource:0
matmul_readvariableop_resource:22
 matmul_1_readvariableop_resource:
identity

identity_1¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp¢ReadVariableOpf
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0_
unstackUnpackReadVariableOp:value:0*
T0* 
_output_shapes
::*	
numt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ£
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_splitx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿ\
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÆ
split_1SplitVBiasAdd_1:output:0Const:output:0split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split`
addAddV2split:output:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
add_1AddV2split:output:1split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mulMulSigmoid_1:y:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
add_2AddV2split:output:2mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿU
mul_1MulSigmoid:y:0states_0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Y
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mulsub:z:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
IdentityIdentity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_1Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ2:ÿÿÿÿÿÿÿÿÿ: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0
Ú
ª
while_cond_108497
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_14
0while_while_cond_108497___redundant_placeholder04
0while_while_cond_108497___redundant_placeholder14
0while_while_cond_108497___redundant_placeholder24
0while_while_cond_108497___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :ÿÿÿÿÿÿÿÿÿ2: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:

_output_shapes
: :

_output_shapes
:
M

A__inference_gru_9_layer_call_and_return_conditional_losses_111393

inputs5
#gru_cell_27_readvariableop_resource:<
*gru_cell_27_matmul_readvariableop_resource:2>
,gru_cell_27_matmul_1_readvariableop_resource:
identity¢!gru_cell_27/MatMul/ReadVariableOp¢#gru_cell_27/MatMul_1/ReadVariableOp¢gru_cell_27/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ2   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
shrink_axis_mask~
gru_cell_27/ReadVariableOpReadVariableOp#gru_cell_27_readvariableop_resource*
_output_shapes

:*
dtype0w
gru_cell_27/unstackUnpack"gru_cell_27/ReadVariableOp:value:0*
T0* 
_output_shapes
::*	
num
!gru_cell_27/MatMul/ReadVariableOpReadVariableOp*gru_cell_27_matmul_readvariableop_resource*
_output_shapes

:2*
dtype0
gru_cell_27/MatMulMatMulstrided_slice_2:output:0)gru_cell_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAddBiasAddgru_cell_27/MatMul:product:0gru_cell_27/unstack:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÇ
gru_cell_27/splitSplit$gru_cell_27/split/split_dim:output:0gru_cell_27/BiasAdd:output:0*
T0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
#gru_cell_27/MatMul_1/ReadVariableOpReadVariableOp,gru_cell_27_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0
gru_cell_27/MatMul_1MatMulzeros:output:0+gru_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/BiasAdd_1BiasAddgru_cell_27/MatMul_1:product:0gru_cell_27/unstack:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
gru_cell_27/ConstConst*
_output_shapes
:*
dtype0*!
valueB"      ÿÿÿÿh
gru_cell_27/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
gru_cell_27/split_1SplitVgru_cell_27/BiasAdd_1:output:0gru_cell_27/Const:output:0&gru_cell_27/split_1/split_dim:output:0*
T0*

Tlen0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split
gru_cell_27/addAddV2gru_cell_27/split:output:0gru_cell_27/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
gru_cell_27/SigmoidSigmoidgru_cell_27/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/add_1AddV2gru_cell_27/split:output:1gru_cell_27/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
gru_cell_27/Sigmoid_1Sigmoidgru_cell_27/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mulMulgru_cell_27/Sigmoid_1:y:0gru_cell_27/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
gru_cell_27/add_2AddV2gru_cell_27/split:output:2gru_cell_27/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
gru_cell_27/ReluRelugru_cell_27/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
gru_cell_27/mul_1Mulgru_cell_27/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
gru_cell_27/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
gru_cell_27/subSubgru_cell_27/sub/x:output:0gru_cell_27/Sigmoid:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gru_cell_27/mul_2Mulgru_cell_27/sub:z:0gru_cell_27/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
gru_cell_27/add_3AddV2gru_cell_27/mul_1:z:0gru_cell_27/mul_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0#gru_cell_27_readvariableop_resource*gru_cell_27_matmul_readvariableop_resource,gru_cell_27_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_111304*
condR
while_cond_111303*8
output_shapes'
%: : : : :ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
NoOpNoOp"^gru_cell_27/MatMul/ReadVariableOp$^gru_cell_27/MatMul_1/ReadVariableOp^gru_cell_27/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2: : : 2F
!gru_cell_27/MatMul/ReadVariableOp!gru_cell_27/MatMul/ReadVariableOp2J
#gru_cell_27/MatMul_1/ReadVariableOp#gru_cell_27/MatMul_1/ReadVariableOp28
gru_cell_27/ReadVariableOpgru_cell_27/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs


d
E__inference_dropout_8_layer_call_and_return_conditional_losses_110737

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>ª
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ2:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
 
_user_specified_nameinputs"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¶
serving_default¢
G
gru_8_input8
serving_default_gru_8_input:0ÿÿÿÿÿÿÿÿÿ;
dense_40
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ô¬
õ
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ã
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
cell

state_spec
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¥
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
»

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
¬
3iter
	4decay
5learning_rate
6momentum
7rho	+rmsz	,rms{	8rms|	9rms}	:rms~	;rms
<rms
=rms"
	optimizer
X
80
91
:2
;3
<4
=5
+6
,7"
trackable_list_wrapper
X
80
91
:2
;3
<4
=5
+6
,7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2ÿ
-__inference_sequential_4_layer_call_fn_108805
-__inference_sequential_4_layer_call_fn_109376
-__inference_sequential_4_layer_call_fn_109397
-__inference_sequential_4_layer_call_fn_109299À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_4_layer_call_and_return_conditional_losses_109707
H__inference_sequential_4_layer_call_and_return_conditional_losses_110031
H__inference_sequential_4_layer_call_and_return_conditional_losses_109324
H__inference_sequential_4_layer_call_and_return_conditional_losses_109349À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÐBÍ
!__inference__wrapped_model_107752gru_8_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
Cserving_default"
signature_map
Ñ

8kernel
9recurrent_kernel
:bias
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
û2ø
&__inference_gru_8_layer_call_fn_110065
&__inference_gru_8_layer_call_fn_110076
&__inference_gru_8_layer_call_fn_110087
&__inference_gru_8_layer_call_fn_110098Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_gru_8_layer_call_and_return_conditional_losses_110251
A__inference_gru_8_layer_call_and_return_conditional_losses_110404
A__inference_gru_8_layer_call_and_return_conditional_losses_110557
A__inference_gru_8_layer_call_and_return_conditional_losses_110710Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
*__inference_dropout_8_layer_call_fn_110715
*__inference_dropout_8_layer_call_fn_110720´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_8_layer_call_and_return_conditional_losses_110725
E__inference_dropout_8_layer_call_and_return_conditional_losses_110737´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ

;kernel
<recurrent_kernel
=bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

[states
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
û2ø
&__inference_gru_9_layer_call_fn_110748
&__inference_gru_9_layer_call_fn_110759
&__inference_gru_9_layer_call_fn_110770
&__inference_gru_9_layer_call_fn_110781Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
A__inference_gru_9_layer_call_and_return_conditional_losses_110934
A__inference_gru_9_layer_call_and_return_conditional_losses_111087
A__inference_gru_9_layer_call_and_return_conditional_losses_111240
A__inference_gru_9_layer_call_and_return_conditional_losses_111393Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
2
*__inference_dropout_9_layer_call_fn_111398
*__inference_dropout_9_layer_call_fn_111403´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_9_layer_call_and_return_conditional_losses_111408
E__inference_dropout_9_layer_call_and_return_conditional_losses_111420´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 :2dense_4/kernel
:2dense_4/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
Ò2Ï
(__inference_dense_4_layer_call_fn_111429¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_4_layer_call_and_return_conditional_losses_111439¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
+:)	2gru_8/gru_cell_26/kernel
5:3	22"gru_8/gru_cell_26/recurrent_kernel
):'	2gru_8/gru_cell_26/bias
*:(22gru_9/gru_cell_27/kernel
4:22"gru_9/gru_cell_27/recurrent_kernel
(:&2gru_9/gru_cell_27/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
k0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÏBÌ
$__inference_signature_wrapper_110054gru_8_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5
80
91
:2"
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
 2
,__inference_gru_cell_26_layer_call_fn_111453
,__inference_gru_cell_26_layer_call_fn_111467¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_111506
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_111545¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
;0
<1
=2"
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 2
,__inference_gru_cell_27_layer_call_fn_111559
,__inference_gru_cell_27_layer_call_fn_111573¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_111612
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_111651¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	vtotal
	wcount
x	variables
y	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
v0
w1"
trackable_list_wrapper
-
x	variables"
_generic_user_object
*:(2RMSprop/dense_4/kernel/rms
$:"2RMSprop/dense_4/bias/rms
5:3	2$RMSprop/gru_8/gru_cell_26/kernel/rms
?:=	22.RMSprop/gru_8/gru_cell_26/recurrent_kernel/rms
3:1	2"RMSprop/gru_8/gru_cell_26/bias/rms
4:222$RMSprop/gru_9/gru_cell_27/kernel/rms
>:<2.RMSprop/gru_9/gru_cell_27/recurrent_kernel/rms
2:02"RMSprop/gru_9/gru_cell_27/bias/rms
!__inference__wrapped_model_107752w:89=;<+,8¢5
.¢+
)&
gru_8_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_4_layer_call_and_return_conditional_losses_111439\+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_4_layer_call_fn_111429O+,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
E__inference_dropout_8_layer_call_and_return_conditional_losses_110725d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ2
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 ­
E__inference_dropout_8_layer_call_and_return_conditional_losses_110737d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ2
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 
*__inference_dropout_8_layer_call_fn_110715W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ2
p 
ª "ÿÿÿÿÿÿÿÿÿ2
*__inference_dropout_8_layer_call_fn_110720W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ2
p
ª "ÿÿÿÿÿÿÿÿÿ2¥
E__inference_dropout_9_layer_call_and_return_conditional_losses_111408\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
E__inference_dropout_9_layer_call_and_return_conditional_losses_111420\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_dropout_9_layer_call_fn_111398O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ}
*__inference_dropout_9_layer_call_fn_111403O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÐ
A__inference_gru_8_layer_call_and_return_conditional_losses_110251:89O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
 Ð
A__inference_gru_8_layer_call_and_return_conditional_losses_110404:89O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
 ¶
A__inference_gru_8_layer_call_and_return_conditional_losses_110557q:89?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 ¶
A__inference_gru_8_layer_call_and_return_conditional_losses_110710q:89?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ2
 §
&__inference_gru_8_layer_call_fn_110065}:89O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2§
&__inference_gru_8_layer_call_fn_110076}:89O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
&__inference_gru_8_layer_call_fn_110087d:89?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ2
&__inference_gru_8_layer_call_fn_110098d:89?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ2Â
A__inference_gru_9_layer_call_and_return_conditional_losses_110934}=;<O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
A__inference_gru_9_layer_call_and_return_conditional_losses_111087}=;<O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
A__inference_gru_9_layer_call_and_return_conditional_losses_111240m=;<?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ2

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ²
A__inference_gru_9_layer_call_and_return_conditional_losses_111393m=;<?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ2

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
&__inference_gru_9_layer_call_fn_110748p=;<O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_gru_9_layer_call_fn_110759p=;<O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_gru_9_layer_call_fn_110770`=;<?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ2

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
&__inference_gru_9_layer_call_fn_110781`=;<?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ2

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_111506·:89\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ2
$!

0/1/0ÿÿÿÿÿÿÿÿÿ2
 
G__inference_gru_cell_26_layer_call_and_return_conditional_losses_111545·:89\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ2
$!

0/1/0ÿÿÿÿÿÿÿÿÿ2
 Ú
,__inference_gru_cell_26_layer_call_fn_111453©:89\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ2
"

1/0ÿÿÿÿÿÿÿÿÿ2Ú
,__inference_gru_cell_26_layer_call_fn_111467©:89\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ2
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ2
"

1/0ÿÿÿÿÿÿÿÿÿ2
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_111612·=;<\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ2
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ
$!

0/1/0ÿÿÿÿÿÿÿÿÿ
 
G__inference_gru_cell_27_layer_call_and_return_conditional_losses_111651·=;<\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ2
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "R¢O
H¢E

0/0ÿÿÿÿÿÿÿÿÿ
$!

0/1/0ÿÿÿÿÿÿÿÿÿ
 Ú
,__inference_gru_cell_27_layer_call_fn_111559©=;<\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ2
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p 
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ
"

1/0ÿÿÿÿÿÿÿÿÿÚ
,__inference_gru_cell_27_layer_call_fn_111573©=;<\¢Y
R¢O
 
inputsÿÿÿÿÿÿÿÿÿ2
'¢$
"
states/0ÿÿÿÿÿÿÿÿÿ
p
ª "D¢A

0ÿÿÿÿÿÿÿÿÿ
"

1/0ÿÿÿÿÿÿÿÿÿ¿
H__inference_sequential_4_layer_call_and_return_conditional_losses_109324s:89=;<+,@¢=
6¢3
)&
gru_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
H__inference_sequential_4_layer_call_and_return_conditional_losses_109349s:89=;<+,@¢=
6¢3
)&
gru_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
H__inference_sequential_4_layer_call_and_return_conditional_losses_109707n:89=;<+,;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
H__inference_sequential_4_layer_call_and_return_conditional_losses_110031n:89=;<+,;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_4_layer_call_fn_108805f:89=;<+,@¢=
6¢3
)&
gru_8_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_109299f:89=;<+,@¢=
6¢3
)&
gru_8_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_109376a:89=;<+,;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_109397a:89=;<+,;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¯
$__inference_signature_wrapper_110054:89=;<+,G¢D
¢ 
=ª:
8
gru_8_input)&
gru_8_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ