// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.Trainers.XGBoost
{
    public abstract class XGBoostTrainerBase<TOptions, TOutput, TTransformer, TModel> : TrainerEstimatorBaseWithGroupId<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class // IPredictorProducing<float>
        where TOptions : XGBoostTrainerBase<TOptions, TOutput, TTransformer, TModel>.OptionsBase, new()
#if false
        : ITrainer<XGBoostModelParameters>,
        ITrainerEstimator<BinaryPredictionTransformer<XGBoostModelParameters>, XGBoostModelParameters>
#endif
    {
#if false
        internal const string LoadNameValue = "XGBoostPredictor";
        internal const string UserNameValue = "XGBoost Predictor";
        internal const string Summary = "The base logic for all XGBoost-based trainers.";

        /// <summary>
        /// The shrinkage rate for trees, used to prevent over-fitting.
	/// Also aliased to "eta"
        /// </summary>
        /// <value>
        /// Valid range is (0,1].
        /// </value>
        [Argument(ArgumentType.AtMostOnce,
            HelpText = "Shrinkage rate for trees, used to prevent over-fitting. Range: (0,1].",
            SortOrder = 2, ShortName = "lr", NullName = "<Auto>")]
        public double? LearningRate;

        /// <summary>
        /// The maximum number of leaves in one tree.
        /// </summary>
        [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum leaves for trees.",
            SortOrder = 2, ShortName = "nl", NullName = "<Auto>")]
        public int? NumberOfLeaves;

        /// <summary>
        /// Minimum loss reduction required to make a further partition on a leaf node of
	/// the tree. The larger gamma is, the more conservative the algorithm will be.
	/// aka: gamma
	/// range: [0,\infnty]
        /// </summary>
	public int? MinSplitLoss;
#endif

        /// <summary>
        /// Maximum depth of a tree. Increasing this value will make the model more complex and
        /// more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively
        /// consumes memory when training a deep tree. exact tree method requires non-zero value.
        /// range: [0,\infnty], default=6
        /// </summary>
        public int? MaxDepth;

        /// <summary>
        /// Minimum sum of instance weight (hessian) needed in a child. If the tree partition step
        /// results in a leaf node with the sum of instance weight less than min_child_weight, then
        /// the building process will give up further partitioning. In linear regression task, this
        /// simply corresponds to minimum number of instances needed to be in each node. The larger
        /// <cref>MinChildWeight</cref> is, the more conservative the algorithm will be.
        /// range: [0,\infnty]
        /// </summary>
        public float? MinChildWeight;

        private protected XGBoostTrainerBase(IHost host,
            SchemaShape.Column feature,
            SchemaShape.Column label, SchemaShape.Column weight = default, SchemaShape.Column groupId = default) : base(host, feature, label, weight, groupId)
        {
        }

#if false
        /// <summary>
        /// L2 regularization term on weights. Increasing this value will make model more conservative
        /// </summary>
        public float? L2Regularization;

        /// <summary>
	/// L1 regularization term on weights. Increasing this value will make model more conservative.
	/// </summary>
        public float? L1Regularization;
#endif

        public class OptionsBase : TrainerInputBaseWithGroupId
        {

            // Static override name map that maps friendly names to XGBMArguments arguments.
            // If an argument is not here, then its name is identical to a lightGBM argument
            // and does not require a mapping, for example, Subsample.
            // For a complete list, see https://xgboost.readthedocs.io/en/latest/parameter.html
            private protected static Dictionary<string, string> NameMapping = new Dictionary<string, string>()
            {
#if false
               {nameof(MinSplitLoss),                         "min_split_loss"},
               {nameof(NumberOfLeaves),                       "num_leaves"},
#endif
	           {nameof(MaxDepth),                             "max_depth" },
               {nameof(MinChildWeight),                   "min_child_weight" },
#if false
    	       {nameof(L2Regularization),          	      "lambda" },
       	       {nameof(L1Regularization),          	      "alpha" }
#endif
            };


#if false
            private protected string GetOptionName(string name)
            {
                if (NameMapping.ContainsKey(name))
                    return NameMapping[name];
                //return XGBoostInterfaceUtils.GetOptionName(name);
		return "";
            }
#endif
        }

        private protected override TModel TrainModelCore(TrainContext context)
        {
#if true
            return null;
#else
            InitializeBeforeTraining();

            Host.CheckValue(context, nameof(context));

            Dataset dtrain = null;
            Dataset dvalid = null;
            CategoricalMetaData catMetaData;
            try
            {
                using (var ch = Host.Start("Loading data for XGBoost"))
                {
                    using (var pch = Host.StartProgressChannel("Loading data for XGBoost"))
                    {
                        dtrain = LoadTrainingData(ch, context.TrainingSet, out catMetaData);
                        if (context.ValidationSet != null)
                            dvalid = LoadValidationData(ch, dtrain, context.ValidationSet, catMetaData);
                    }
                }
                using (var ch = Host.Start("Training with XGBoost"))
                {
                    using (var pch = Host.StartProgressChannel("Training with XGBoost"))
                        TrainCore(ch, pch, dtrain, catMetaData, dvalid);
                }
            }
            finally
            {
                dtrain?.Dispose();
                dvalid?.Dispose();
                DisposeParallelTraining();
            }
            return CreatePredictor();
#endif
        }

#if false

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: false, caching: false);

        /// <summary>
        /// Auxiliary information about the trainer in terms of its capabilities
        /// and requirements.
        /// </summary>
        public TrainerInfo Info => _info;

        internal XGBoostTrainer(IHostEnvironment env, Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadNameValue);
            _host.CheckValue(options, nameof(options));
        }

        /// <summary>
        /// Initializes XGBoostTrainer object.
        /// </summary>
        internal XGBoostTrainer(IHostEnvironment env, String labelColumn, String weightColunn = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadNameValue);
            _host.CheckValue(labelColumn, nameof(labelColumn));
            _host.CheckValueOrNull(weightColunn);

            _labelColumnName = labelColumn;
            _weightColumnName = weightColunn != null ? weightColunn : null;
        }

        /// <summary>
        /// Trains and returns a <see cref="BinaryPredictionTransformer{XGBoostModelParameters}"/>.
        /// </summary>
        public BinaryPredictionTransformer<XGBoostModelParameters> Fit(IDataView input)
        {
            RoleMappedData trainRoles = new RoleMappedData(input, label: _labelColumnName, feature: null, weight: _weightColumnName);
            var pred = ((ITrainer<XGBoostModelParameters>)this).Train(new TrainContext(trainRoles));
            return new BinaryPredictionTransformer<XGBoostModelParameters>(_host, pred, input.Schema, featureColumn: null, labelColumn: _labelColumnName);
        }

        private XGBoostModelParameters Train(TrainContext context)
        {
            _host.CheckValue(context, nameof(context));
            var data = context.TrainingSet;
            data.CheckBinaryLabel();
            _host.CheckParam(data.Schema.Label.HasValue, nameof(data), "Missing Label column");
            var labelCol = data.Schema.Label.Value;
            _host.CheckParam(labelCol.Type == BooleanDataViewType.Instance, nameof(data), "Invalid type for Label column");

            double pos = 0;
            double neg = 0;

            int colWeight = -1;
            if (data.Schema.Weight?.Type == NumberDataViewType.Single)
                colWeight = data.Schema.Weight.Value.Index;

            var cols = colWeight > -1 ? new DataViewSchema.Column[] { labelCol, data.Schema.Weight.Value } : new DataViewSchema.Column[] { labelCol };

            using (var cursor = data.Data.GetRowCursor(cols))
            {
                var getLab = cursor.GetGetter<bool>(data.Schema.Label.Value);
                var getWeight = colWeight >= 0 ? cursor.GetGetter<float>(data.Schema.Weight.Value) : null;
                bool lab = default;
                float weight = 1;
                while (cursor.MoveNext())
                {
                    getLab(ref lab);
                    if (getWeight != null)
                    {
                        getWeight(ref weight);
                        if (!(0 < weight && weight < float.PositiveInfinity))
                            continue;
                    }

                    // Testing both directions effectively ignores NaNs.
                    if (lab)
                        pos += weight;
                    else
                        neg += weight;
                }
            }

            float prob = prob = pos + neg > 0 ? (float)(pos / (pos + neg)) : float.NaN;
            return new XGBoostModelParameters(_host, prob);
        }

        IPredictor ITrainer.Train(TrainContext context) => Train(context);

        XGBoostModelParameters ITrainer<XGBoostModelParameters>.Train(TrainContext context) => Train(context);

        private static SchemaShape.Column MakeFeatureColumn(string featureColumn)
            => new SchemaShape.Column(featureColumn, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Single, false);

        private static SchemaShape.Column MakeLabelColumn(string labelColumn)
            => new SchemaShape.Column(labelColumn, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false);

        /// <summary>
        /// Returns the <see cref="SchemaShape"/> of the schema which will be produced by the transformer.
        /// Used for schema propagation and verification in a pipeline.
        /// </summary>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            var outColumns = inputSchema.ToDictionary(x => x.Name);

            var newColumns = new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation())),
                new SchemaShape.Column(DefaultColumnNames.Probability, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation(true))),
                new SchemaShape.Column(DefaultColumnNames.PredictedLabel, SchemaShape.Column.VectorKind.Scalar, BooleanDataViewType.Instance, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
            foreach (SchemaShape.Column column in newColumns)
                outColumns[column.Name] = column;

            return new SchemaShape(outColumns.Values);
        }
#endif
        private protected abstract TModel CreatePredictor();
    }
}
