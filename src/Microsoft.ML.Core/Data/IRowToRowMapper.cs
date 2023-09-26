﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Data;

/// <summary>
/// This interface maps an input <see cref="DataViewRow"/> to an output <see cref="DataViewRow"/>. Typically, the output contains
/// both the input columns and new columns added by the implementing class, although some implementations may
/// return a subset of the input columns.
/// This interface is similar to <see cref="ISchemaBoundRowMapper"/>, except it does not have any input role mappings,
/// so to rebind, the same input column names must be used.
/// Implementations of this interface are typically created over defined input <see cref="DataViewSchema"/>.
/// </summary>
public interface IRowToRowMapper
{
    /// <summary>
    /// Mappers are defined as accepting inputs with this very specific schema.
    /// </summary>
    DataViewSchema InputSchema { get; }

    /// <summary>
    /// Gets an instance of <see cref="DataViewSchema"/> which describes the columns' names and types in the output generated by this mapper.
    /// </summary>
    DataViewSchema OutputSchema { get; }

    /// <summary>
    /// Given a set of columns, return the input columns that are needed to generate those output columns.
    /// </summary>
    IEnumerable<DataViewSchema.Column> GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns);

    /// <summary>
    /// Get an <see cref="DataViewRow"/> with the indicated active columns, based on the input <paramref name="input"/>.
    /// Getting values on inactive columns of the returned row will throw.
    ///
    /// The <see cref="DataViewRow.Schema"/> of <paramref name="input"/> should be the same object as
    /// <see cref="InputSchema"/>. Implementors of this method should throw if that is not the case. Conversely,
    /// the returned value must have the same schema as <see cref="OutputSchema"/>.
    ///
    /// This method creates a live connection between the input <see cref="DataViewRow"/> and the output <see
    /// cref="DataViewRow"/>. In particular, when the getters of the output <see cref="DataViewRow"/> are invoked, they invoke the
    /// getters of the input row and base the output values on the current values of the input <see cref="DataViewRow"/>.
    /// The output <see cref="DataViewRow"/> values are re-computed when requested through the getters. Also, the returned
    /// <see cref="DataViewRow"/> will dispose <paramref name="input"/> when it is disposed.
    /// </summary>
    DataViewRow GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns);
}
