// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_
#define IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_

#include "llvm/ADT/StringMap.h"

namespace mlir::iree_compiler {

class EmbeddedDataDirectory {
public:
  static EmbeddedDataDirectory &getGlobal() {
    static EmbeddedDataDirectory global;
    return global;
  }

  bool addFile(llvm::StringRef fileName, llvm::StringRef contents) {
    auto [iter, success] = directory.insert({fileName, contents});
    return success;
  }

  std::optional<llvm::StringRef> getFile(llvm::StringRef fileName) const {
    auto iter = directory.find(fileName);
    if (iter == directory.end()) {
      return std::nullopt;
    }
    return iter->getValue();
  }

private:
  llvm::StringMap<llvm::StringRef> directory;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_EMBEDDEDDATADIRECTORY_H_
