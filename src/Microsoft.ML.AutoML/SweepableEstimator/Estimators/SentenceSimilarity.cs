﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using Microsoft.ML.TorchSharp;

namespace Microsoft.ML.AutoML.CodeGen
{
    internal partial class SentenceSimilarityRegression
    {
        public override IEstimator<ITransformer> BuildFromOption(MLContext context, SentenceSimilarityOption param)
        {
            return context.Regression.Trainers.SentenceSimilarity(
                    labelColumnName: param.LabelColumnName,
                    sentence1ColumnName: param.Sentence1ColumnName,
                    scoreColumnName: param.ScoreColumnName,
                    sentence2ColumnName: param.Sentence2ColumnName,
                    batchSize: param.BatchSize,
                    maxEpochs: param.MaxEpochs,
                    architecture: param.Architecture);
        }
    }
}
