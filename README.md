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

Feel free to report what other operators might be useful (along with example models) via the Issues tab.

The tool works from command line or from the browser.

## Architecture

The models are loaded using TensorFlow.js library.
Each layer is first compiled separately, and the generated code is run in simulation
(a JavaScript function is generated, where each line corresponds to a single assembly instruction).
The results are compared with running the same layer in TensorFlow.js.
This process can be disabled with `--no-validate` option.
Then layers are composed and the final binary code is generated.
The binary is position-independent and can be loaded from any word-aligned address in flash or RAM.

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
