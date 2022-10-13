// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Model.OnnxConverter;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;

namespace Microsoft.ML.Trainers.XGBoost
{
    internal static class Defaults
    {
        public const int NumberOfIterations = 100;
    }

#if false
    public sealed class XGBoostBinaryClassificationTransformer : OneToOneTransformerBase
    {
	private Booster _booster;
	private int _numColumns;
	
        internal XGBoostBinaryClassificationTransformer(IHost host, Booster booster, params (string outputColumnName, string inputColumnName)[] columns) : base(host, columns)
        {
	  _booster = booster;
	  _numColumns = columns.Length;
        }

        internal XGBoostBinaryClassificationTransformer(IHost host, ModelLoadContext ctx) : base(host, ctx)
        {
        }

        private protected override IRowMapper MakeRowMapper(DataViewSchema schema) => new Mapper(this, schema);

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            Host.CheckValue(ctx, nameof(ctx));
        }

        private sealed class Mapper : OneToOneMapperBase, ISaveAsOnnx
        {
            private readonly XGBoostBinaryClassificationTransformer _parent;
            private readonly int _numColumns;
            public Mapper(XGBoostBinaryClassificationTransformer parent, DataViewSchema inputSchema)
                : base(parent.Host.Register(nameof(Mapper)), parent, inputSchema)
            {
                _parent = parent;
		_numColumns = _parent._numColumns;
            }

            public bool CanSaveOnnx(OnnxContext ctx) => true;

            public void SaveAsOnnx(OnnxContext ctx)
            {
                throw new NotImplementedException();
            }

            protected override DataViewSchema.DetachedColumn[] GetOutputColumnsCore()
            {
	        var result = new DataViewSchema.DetachedColumn[_numColumns];
                for (int i = 0; i < _numColumns; i++)
                    result[i] = new DataViewSchema.DetachedColumn("PredictedLabel", NumberDataViewType.Int16, null);
                return result;
            }

            protected override Delegate MakeGetter(DataViewRow input, int iinfo, Func<int, bool> activeOutput, out Action disposer)
            {
		Contracts.AssertValue(input);
                Contracts.Assert(0 <= iinfo && iinfo < _numColumns);
                disposer = null;

                var srcGetter = input.GetGetter<VBuffer<float>>(input.Schema[ColMapNewToOld[iinfo]]);
                var src = default(VBuffer<float>);

                ValueGetter<VBuffer<float>> dstGetter = (ref VBuffer<float> dst) =>
                    {
                        srcGetter(ref src);
                        Predict(Host, in src, ref dst);
                    };

                return dstGetter;
            }

	    private void Predict(IExceptionContext ectx, in VBuffer<float> src, ref VBuffer<float> dst)
            {
		dst = _parent._booster.Predict(src);
            }

        }
    }

    public sealed class XGBoostBinaryClassificationEstimator : IEstimator<XGBoostBinaryClassificationTransformer>
    {
        private readonly IHost _host;

       public sealed class Options : TrainerInputBase
       {
            /// <summary>
            /// Maximum tree depth for base learners
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum tree depth for base learners.", ShortName = "us")]
            public int MaxDepth = 3;
       }

        public XGBoostBinaryClassificationEstimator(IHost host, XGBoostBinaryClassificationTransformer transformer)         {
            _host = Contracts.CheckRef(host, nameof(host)).Register(nameof(XGBoostBinaryClassificationEstimator));
        }

        public XGBoostBinaryClassificationEstimator(IHostEnvironment env, string labelColumnName, string featureColumnName, int? numberOfLeaves, int? minimumExampleCountPerLeaf, double? learningRate, int? numberOfIterations)
        {
            _host = Contracts.CheckRef(env, nameof(env)).Register(nameof(XGBoostBinaryClassificationEstimator));
        }

        public XGBoostBinaryClassificationTransformer Fit(IDataView input)
        {
	    	var featuresColumn = input.Schema["Features"];
		var labelColumn = input.Schema["Label"];
		int featureDimensionality = default(int);
		if (featuresColumn.Type is VectorDataViewType vt) {
		  featureDimensionality = vt.Size;
		} else {
		  _host.Except($"A vector input is expected");
		}
		int samples = 0;
		int maxSamples = 10000;

		float[] data = new float[ maxSamples * featureDimensionality];
		float[] dataLabels = new float[ maxSamples ];
		Span<float> dataSpan = new Span<float>(data);

		using (var cursor = input.GetRowCursor(new[] { featuresColumn, labelColumn })) {

		  float labelValue = default;
		  VBuffer<float> featureValues = default(VBuffer<float>);

		  var featureGetter = cursor.GetGetter< VBuffer<float> >(featuresColumn);
		  var labelGetter = cursor.GetGetter<float>(labelColumn);

		  while (cursor.MoveNext() && samples < maxSamples) {
		    featureGetter(ref featureValues);
		    labelGetter(ref labelValue);

		    int offset = samples * featureDimensionality;
		    Span<float> target = dataSpan.Slice(offset, featureDimensionality);
		    featureValues.GetValues().CopyTo(target);
		    dataLabels[samples] = labelValue;
		    samples++;
	    	  }

		  DMatrix trainMat = new DMatrix(data, (uint)maxSamples, (uint)featureDimensionality, dataLabels);
		  Booster booster = new Booster(trainMat);
  		  return new XGBoostBinaryClassificationTransformer(_host, booster, ("Features", "PredictedLabel"));
	  	}
        }

#if true
        // Used for schema propagation and verification in a pipeline (i.e., in an Estimator chain).
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
#else
        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));
            return new SchemaShape(inputSchema);
        }
#endif
    }
#endif

    /// <summary>
    /// Model parameters for <see cref="XGBoostBinaryTrainer"/>.
    /// </summary>
    public sealed class XGBoostBinaryModelParameters : TreeEnsembleModelParametersBasedOnRegressionTree
    {
        internal const string LoaderSignature = "XGBoostExec";
        internal const string RegistrationName = "XGBoostPredictor";
        private static VersionInfo GetVersionInfo()
        {
            // REVIEW: can we decouple the version from FastTree predictor version ?
            return new VersionInfo(
                modelSignature: "XGBBINCL",
                // verWrittenCur: 0x00010001, // Initial
                // verWrittenCur: 0x00010002, // _numFeatures serialized
                // verWrittenCur: 0x00010003, // Ini content out of predictor
                //verWrittenCur: 0x00010004, // Add _defaultValueForMissing
                verWrittenCur: 0x00010001,
                verReadableCur: 0x00010004,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(XGBoostBinaryModelParameters).Assembly.FullName);
        }

        internal XGBoostBinaryModelParameters(IHostEnvironment env, InternalTreeEnsemble trainedEnsemble, int featureCount, string innerArgs)
            : base(env, RegistrationName, trainedEnsemble, featureCount, innerArgs)
        {
        }

        private XGBoostBinaryModelParameters(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, RegistrationName, ctx, GetVersionInfo())
        {
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            ctx.SetVersionInfo(GetVersionInfo());
        }

#if false
        internal static IPredictorProducing<float> Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());
            var predictor = new LightGbmBinaryModelParameters(env, ctx);
            ICalibrator calibrator;
            ctx.LoadModelOrNull<ICalibrator, SignatureLoadModel>(env, out calibrator, @"Calibrator");
            if (calibrator == null)
                return predictor;
            return new ValueMapperCalibratedModelParameters<LightGbmBinaryModelParameters, ICalibrator>(env, predictor, calibrator);
        }
#endif
        private protected override uint VerNumFeaturesSerialized { get { return 0x00010002; } }

        private protected override uint VerDefaultValueSerialized => 0x00010004;

        private protected override uint VerCategoricalSplitSerialized => 0x00010005;

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;
    }

    /// <summary>
    /// The <see cref="IEstimator{TTransformer}"/> for training a boosted decision tree binary classification model using XGBoost.
    /// </summary>
    public sealed class XGBoostBinaryTrainer : XGBoostTrainerBase<XGBoostBinaryTrainer.Options, float, BinaryPredictionTransformer<XGBoostBinaryModelParameters>, XGBoostBinaryModelParameters>
    {
        internal const string UserName = "XGBoost Binary Classifier";
        internal const string LoadNameValue = "XGBoostBinary";
        internal const string ShortName = "XGBoost";
        internal const string Summary = "Train a XGBoost binary classification model.";

        public XGBoostBinaryTrainer(IHost host, SchemaShape.Column feature, SchemaShape.Column label, SchemaShape.Column weight = default, SchemaShape.Column groupId = default) : base(host, feature, label, weight, groupId)
        {
        }

        public override TrainerInfo Info => throw new NotImplementedException();

        private protected override PredictionKind PredictionKind => PredictionKind.BinaryClassification;

        private protected override XGBoostBinaryModelParameters CreatePredictor()
        {
            throw new NotImplementedException();
        }

        private protected override SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema)
        {
            return new[]
            {
                new SchemaShape.Column(DefaultColumnNames.Score, SchemaShape.Column.VectorKind.Scalar, NumberDataViewType.Single, false, new SchemaShape(AnnotationUtils.GetTrainerOutputAnnotation()))
            };
        }
        private protected override BinaryPredictionTransformer<XGBoostBinaryModelParameters> MakeTransformer(XGBoostBinaryModelParameters model, DataViewSchema trainSchema)
            => new BinaryPredictionTransformer<XGBoostBinaryModelParameters>(Host, model, trainSchema, FeatureColumn.Name);

        public sealed class Options : OptionsBase
        {

            public enum EvaluateMetricType
            {
                None,
                Default,
                Logloss,
                Error,
                AreaUnderCurve,
            };

#if false
            /// <summary>
            /// Whether training data is unbalanced.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use for binary classification when training data is not balanced.", ShortName = "us")]
            public bool UnbalancedSets = false;

            /// <summary>
            /// Controls the balance of positive and negative weights in <see cref="LightGbmBinaryTrainer"/>.
            /// </summary>
            /// <value>
            /// This is useful for training on unbalanced data. A typical value to consider is sum(negative cases) / sum(positive cases).
            /// </value>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Control the balance of positive and negative weights, useful for unbalanced classes." +
                " A typical value to consider: sum(negative cases) / sum(positive cases).",
                ShortName = "ScalePosWeight")]
            public double WeightOfPositiveExamples = 1;

            /// <summary>
            /// Parameter for the sigmoid function.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Parameter for the sigmoid function.", ShortName = "sigmoid")]
            [TGUI(Label = "Sigmoid", SuggestedSweeps = "0.5,1")]
            public double Sigmoid = 0.5;

            /// <summary>
            /// Determines what evaluation metric to use.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Evaluation metrics.",
                ShortName = "em")]
            public EvaluateMetricType EvaluationMetric = EvaluateMetricType.Logloss;

            static Options()
            {
                NameMapping.Add(nameof(EvaluateMetricType), "metric");
                NameMapping.Add(nameof(EvaluateMetricType.None), "None");
                NameMapping.Add(nameof(EvaluateMetricType.Default), "");
                NameMapping.Add(nameof(EvaluateMetricType.Logloss), "binary_logloss");
                NameMapping.Add(nameof(EvaluateMetricType.Error), "binary_error");
                NameMapping.Add(nameof(EvaluateMetricType.AreaUnderCurve), "auc");
                NameMapping.Add(nameof(WeightOfPositiveExamples), "scale_pos_weight");
            }

            internal override Dictionary<string, object> ToDictionary(IHost host)
            {
                var res = base.ToDictionary(host);
                res[GetOptionName(nameof(UnbalancedSets))] = UnbalancedSets;
                res[GetOptionName(nameof(WeightOfPositiveExamples))] = WeightOfPositiveExamples;
                res[GetOptionName(nameof(Sigmoid))] = Sigmoid;
                res[GetOptionName(nameof(EvaluateMetricType))] = GetOptionName(EvaluationMetric.ToString());

                return res;
            }
#endif
        }

#if false
        internal LightGbmBinaryTrainer(IHostEnvironment env, Options options)
             : base(env, LoadNameValue, options, TrainerUtils.MakeBoolScalarLabel(options.LabelColumnName))
        {
            Contracts.CheckUserArg(options.Sigmoid > 0, nameof(Options.Sigmoid), "must be > 0.");
            Contracts.CheckUserArg(options.WeightOfPositiveExamples > 0, nameof(Options.WeightOfPositiveExamples), "must be > 0.");
        }

        /// <summary>
        /// Initializes a new instance of <see cref="LightGbmBinaryTrainer"/>
        /// </summary>
        /// <param name="env">The private instance of <see cref="IHostEnvironment"/>.</param>
        /// <param name="labelColumnName">The name of The label column.</param>
        /// <param name="featureColumnName">The name of the feature column.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The number of leaves to use.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points allowed in a leaf of the tree, out of the subsampled data.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        internal LightGbmBinaryTrainer(IHostEnvironment env,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
            : this(env,
                  new Options()
                  {
                      LabelColumnName = labelColumnName,
                      FeatureColumnName = featureColumnName,
                      ExampleWeightColumnName = exampleWeightColumnName,
                      NumberOfLeaves = numberOfLeaves,
                      MinimumExampleCountPerLeaf = minimumExampleCountPerLeaf,
                      LearningRate = learningRate,
                      NumberOfIterations = numberOfIterations
                  })
        {
        }

        private protected override void CheckDataValid(IChannel ch, RoleMappedData data)
        {
            Host.AssertValue(ch);
            base.CheckDataValid(ch, data);
            var labelType = data.Schema.Label.Value.Type;
            if (!(labelType is BooleanDataViewType || labelType is KeyDataViewType || labelType == NumberDataViewType.Single))
            {
                throw ch.ExceptParam(nameof(data),
                    $"Label column '{data.Schema.Label.Value.Name}' is of type '{labelType.RawType}', but must be unsigned int, boolean or float.");
            }
        }

        private protected override void CheckAndUpdateParametersBeforeTraining(IChannel ch, RoleMappedData data, float[] labels, int[] groups)
            => GbmOptions["objective"] = "binary";


        private protected override BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
            MakeTransformer(CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator> model, DataViewSchema trainSchema)
         => new BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>(Host, model, trainSchema, FeatureColumn.Name);
#endif
    }
}
