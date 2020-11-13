# ML4F - Machine Learning model compiler for Cortex-M4F

ML4F takes a [Keras](https://keras.io/) sequential model as an input and compiles it directly to 
ARM Thumb machine code for Cortex-M4F and better (M7, M33 etc.).
The performance (latency) is typically an order of magnitude better than the
[Tensorflow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) interpreter
(with `float32` models).

The input model generally needs to be in [Tensorflow.js](https://www.tensorflow.org/js) format, but the command line tool can
invoke [Python scripts](https://www.tensorflow.org/js/guide/conversion) to convert from `.h5` or `.pb` models.
Once compiled, weights can be stored as `float32` or `float16`.

The following operators are supported:
* `Conv2D`
* `MaxPooling2D`
* `Dense`

Plus some no-ops:
* `InputLayer`
* `Dropout`
* `Flatten`
* `Reshape`

Feel free to report what other operators might be useful (along with example models) via 
[GitHub Issues](https://github.com/microsoft/ml4f/issues).

## Usage

```bash
npm i -g ml4f
ml4f my-model
```

Typical invocation might look like this:

```bash
ml4f           --basename model-float32 my-model.h5
ml4f --float16 --basename model-float16 built/converted.tfjs
```

First line compiles `my-model.h5` using `float32` weights, with results in `built/model-float32.*`.
The second line compiles with `float16` weights, using temporary file created by the first
line to speed things up (Python TensorFlow is really slow to load).
Results are in `built/model-float16.*`.

Run `ml4f --help` for more info.

You can also use it as a library from a browser (in which case it can only take TF.js models).

## Evaluating models

You can pass `--eval test.json` option to evaluated the model on given input data - this will
print confusion matrix and accuracy.
The `test.json` has two fields `x` and `y`. The field `x` contains a batch of input tensors,
and `y` a batch of output tensors, with proper nesting.
For example, for input of shape `2x3` and output of shape `4`:

```json
{ 
  "x": [
    [ [ 0.1, 0.2, -0.3 ], [ 0.2, -0.22, 0 ] ],
    [ [ -0.1, 0.3, 0.1 ], [ 0.32, 0.2, 1 ] ]
  ],
  "y": [
      [ 0, 1, 0, 0 ],
      [ 1, 0, 0, 0 ]
  ]
}
```

If you have data as NumPy arrays, you can use the following snippet to save it as JSON:

```python
import json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open('test.json', 'w') as outfile:
    json.dump({"x": xs_test, "y": ys_test}, outfile, cls=NumpyEncoder)
```

Evaluation stats look like the following:

```
Accuracy: 0.9560
  245    0    1    2
    6   84    4    0
    3    2   73    0
    4    0    0   76

model: 12.75k; code: 2.46k (19.3%); arena: 4.38k; test 0.00k
total cycles: 225149 (2.680ms at 84MHz)
```

## Architecture

The models are loaded using TensorFlow.js library.
Each layer is first compiled separately, and the generated code is run in simulation
(a JavaScript function is generated, where each line corresponds to a single assembly instruction).
The results are compared with running the same layer in TensorFlow.js.
This process can be disabled with `--no-validate` option.
Then layers are composed and the final binary code is generated.

The binary is position-independent and can be loaded from any word-aligned address in flash or RAM.
Look in `sample/` folder for example invocation from C, 
or check out our [MakeCode extension](https://github.com/microsoft/pxt-ml4f).


## Compiling

```
yarn install
yarn watch
# in another window
http-server -c1
```

Then open http://localhost:8080/

Also, run `./ml4f` in this folder.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
