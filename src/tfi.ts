// These types are normally defined in TF.js, but they are not exported for some reason.

import { DataType } from "@tensorflow/tfjs";

export type ActivationIdentifier = 'elu' | 'hardSigmoid' | 'linear' | 'relu' | 'relu6' |
    'selu' | 'sigmoid' | 'softmax' | 'softplus' | 'softsign' | 'tanh';

export type ConstraintIdentifier = 'maxNorm' | 'minMaxNorm' | 'nonNeg' | 'unitNorm' | string;
export type RegularizerIdentifier = 'l1l2' | string;

export declare interface LayerArgs {
    dtype?: DataType;
}

export declare interface DenseLayerArgs extends LayerArgs {
    activation?: ActivationIdentifier;
    units: number;
    useBias?: boolean;
}

export type PaddingMode = 'valid' | 'same' | 'causal';

export type DataFormat = 'channelsFirst' | 'channelsLast';

export declare interface BaseConvLayerArgs extends LayerArgs {
    kernelSize: number[];
    strides: number[];
    padding: PaddingMode;
    dataFormat: DataFormat;

    dilationRate?: number | [number] | [number, number] | [number, number, number];
    activation?: ActivationIdentifier;
    useBias?: boolean;
}

export declare interface ConvLayerArgs extends BaseConvLayerArgs {
    filters: number;
}

export declare interface Pooling2DLayerArgs extends LayerArgs {
    poolSize: [number, number];
    strides: [number, number];
    padding: PaddingMode;
    dataFormat: DataFormat;
}
