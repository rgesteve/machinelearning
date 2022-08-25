﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.OneDal;

namespace Microsoft.ML
{
    public static class OneDalCatalog
    {
        public static KnnClassificationTrainer KnnClassification(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
	    int numClasses)
            => new KnnClassificationTrainer(CatalogUtils.GetEnvironment(catalog), numClasses);
    }
}
