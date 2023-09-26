﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.Data.Analysis
{
    internal static class PrimitiveColumnContainerHelpers
    {
        internal static DataFrameBuffer<T> GetOrCreateMutable<T>(this IList<ReadOnlyDataFrameBuffer<T>> bufferList, int index)
            where T : unmanaged
        {
            var sourceBuffer = bufferList[index];

            if (sourceBuffer is not DataFrameBuffer<T> mutableBuffer)
            {
                mutableBuffer = DataFrameBuffer<T>.GetMutableBuffer(sourceBuffer);
                bufferList[index] = mutableBuffer;
            }

            return mutableBuffer;
        }
    }
}
