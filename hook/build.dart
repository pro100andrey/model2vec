import 'dart:convert';
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;

void main(List<String> args) async {
  await build(args, (input, output) async {
    // An invocation that did not ask for code assets carries no target OS and
    // no architecture, so there is nothing here to build for — and reading
    // `input.config.code` in that mode throws by contract: "HookConfig.code
    // should only be accessed when building code assets."
    //
    // The cost of not asking is out of proportion to the mistake: the throw
    // exits the hook with 255, and the runner then fails the package's whole
    // Dart build, including the invocation that DID ask and did produce the
    // library. `flutter run` reports "Building native assets failed" and never
    // starts the app. Flutter reaches this branch on every run — it invokes
    // hooks a second time to bundle assets (`FlutterHookRunnerNative.runHooks`
    // passes `buildCodeAssets: null`), and with data assets behind a feature
    // flag that is off on stable, that invocation asks for nothing at all.
    // `dart build` never makes it, which is why the failure looks like a
    // Flutter-only one.
    //
    // Returning here leaves an empty output, which is what the runner expects
    // of an invocation that asked for nothing: `validateCodeAssetBuildOutput`
    // short-circuits on the same flag.
    if (!input.config.buildCodeAssets) {
      return;
    }

    final packageRoot = input.packageRoot.toFilePath();
    final nativeDir = p.join(packageRoot, 'native');
    final manifestPath = p.join(nativeDir, 'Cargo.toml');
    final crateName = _parseCrateName(manifestPath);

    // Where cargo builds. The invoker hands out a directory for exactly this —
    // "shared output and intermediate artifacts", unique per hook, with
    // concurrent invocations serialised by the runner — and cargo's default of
    // `<crate>/target` is the one place a hook must not use: the crate lives in
    // the pub cache, which every project on the machine shares.
    //
    // Sharing it is not a tidiness argument, it is the cache. The runner tracks
    // the artifact it was handed, and macOS stamps a fresh `LC_UUID` into every
    // link, so a relink is a content change even when no source moved. Two
    // consumers building the same crate under different configs — `dart test`
    // (`linking_enabled: false`) and `dart build` (`true`) are enough, and a
    // second checkout or a separate Flutter workspace does it too — therefore
    // overwrite each other's artifact and invalidate each other's hook. The
    // runner then deletes `output.json` and re-runs the hook, and the files a
    // parallel `dart test` has already scheduled fail with `No asset with id`.
    // Undiagnosable from the failing run: the build that poisons the cache is
    // never the one that reports it.
    final targetDir = p.join(input.outputDirectoryShared.toFilePath(), 'cargo');

    final codeConfig = input.config.code;

    // We default to release mode
    const buildMode = 'release';
    final targetTriple = _mapToTriple(codeConfig);

    final linkMode = codeConfig.linkModePreference == .static
        ? StaticLinking()
        : DynamicLoadingBundled();

    // 2. Execute Cargo Build
    // Optimization: Don't use --target if it's the host architecture
    // to avoid Cargo path complexity (target/release vs target/triple/release)
    final isHost = _isHost(codeConfig);

    stdout.writeln(
      'Building Rust crate "$crateName" for $targetTriple (isHost: $isHost)...',
    );

    // Add rustup target to ensure cross-compilation support
    if (!isHost) {
      try {
        final rustupResult = await Process.run(
          'rustup',
          ['target', 'add', targetTriple],
          workingDirectory: nativeDir,
        );

        if (rustupResult.exitCode != 0) {
          stdout.writeln(
            'Warning: Failed to add target $targetTriple via rustup '
            '(ignore if rustup is not used):\n${rustupResult.stderr}',
          );
        }
      } on Object catch (e) {
        stdout.writeln(
          'Warning: rustup command not found. Assuming target is managed '
          'externally. $e',
        );
      }
    }

    final env = _getBuildEnvVars(codeConfig);
    final result = await Process.run(
      'cargo',
      [
        'build',
        if (buildMode == 'release') '--release',
        if (!isHost) ...['--target', targetTriple],
        ...['--target-dir', targetDir],
      ],
      workingDirectory: nativeDir,
      environment: env,
    );

    if (result.exitCode != 0) {
      throw Exception('Rust build failed:\n${result.stderr}');
    }

    // 3. Identify Artifact
    final libFileName = _getLibraryFileName(codeConfig.targetOS, crateName);

    final binaryPath = isHost
        ? p.join(targetDir, buildMode, libFileName)
        : p.join(targetDir, targetTriple, buildMode, libFileName);

    if (!File(binaryPath).existsSync()) {
      throw Exception('Artifact not found at $binaryPath');
    }

    // 4. Register Asset
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'model2vec.so',
        linkMode: linkMode,
        file: .file(binaryPath),
      ),
    );

    // 5. Track dependencies precisely using .d file
    //
    // `setExtension`, not `'$binaryPath.d'`. Cargo writes ONE dep-info file per
    // target, named after the target rather than after the artifact: a crate
    // built as `["staticlib", "cdylib"]` produces `libm2v_ffi.a`,
    // `libm2v_ffi.dylib` and a single `libm2v_ffi.d` beside them. Appending to
    // the artifact path asked for `libm2v_ffi.dylib.d`, which cargo never
    // writes on any platform — so the parse below returned nothing EVERY time
    // and the fallback was not a fallback but the only path ever taken.
    //
    // That mattered far more than a missed optimisation. The fallback declared
    // `native/` — and `native/` contains cargo's own `target/`. A hook whose
    // declared input contains its own output invalidates itself: anything
    // written into the build tree, such as a second consumer building the same
    // crate, makes the runner report `File modified during build. Build must
    // be rerun.` and invoke the hook again. Measured here, a file written into
    // `target/` without touching `src/`: 1 re-run and 1.55 s before this fix,
    // 0 and 0.85 s after. The re-run is cheap only while cargo has nothing to
    // do — when it coincides with a real rebuild the cost is the rebuild, and
    // in a dependent workspace that was 68 s against 3.5 s settled, with the
    // spawning tests of a parallel `dart test` timing out. See
    // dart-lang/native#1998 for the same shape reported against
    // `native_toolchain_c`, and dart.dev/tools/hooks: dependencies are the
    // inputs a hook READS, never the outputs it produces.
    final depFilePath = p.setExtension(binaryPath, '.d');
    final deps = _parseDependencies(depFilePath);
    if (deps.isNotEmpty) {
      output.dependencies.addAll(deps);
    } else {
      // Fallback if the .d file is missing — the crate's real inputs, named one
      // by one. Deliberately NOT `nativeDir`: that is the parent of `target/`,
      // and declaring it is the self-invalidating loop described above.
      output.dependencies.add(.directory(p.join(nativeDir, 'src')));
      for (final name in const ['Cargo.lock', 'rust-toolchain.toml']) {
        final file = File(p.join(nativeDir, name));
        if (file.existsSync()) {
          output.dependencies.add(.file(file.path));
        }
      }
    }
    // Always track the manifest
    output.dependencies.add(.file(manifestPath));
  });
}

bool _isHost(CodeConfig codeConfig) {
  final os = codeConfig.targetOS;
  final arch = codeConfig.targetArchitecture;

  if (Platform.isLinux && os == .linux && arch == .x64) {
    return true;
  }

  if (Platform.isMacOS && os == .macOS) {
    return arch == .arm64 || arch == .x64;
  }

  if (Platform.isWindows && os == .windows && arch == .x64) {
    return true;
  }

  return false;
}

String _mapToTriple(CodeConfig codeConfig) {
  final os = codeConfig.targetOS;
  final arch = codeConfig.targetArchitecture;

  return switch ((os, arch)) {
    (.linux, .x64) => 'x86_64-unknown-linux-gnu',
    (.linux, .arm64) => 'aarch64-unknown-linux-gnu',
    (.macOS, .x64) => 'x86_64-apple-darwin',
    (.macOS, .arm64) => 'aarch64-apple-darwin',
    (.windows, .x64) => 'x86_64-pc-windows-msvc',
    (.windows, .arm64) => 'aarch64-pc-windows-msvc',
    (.android, .arm64) => 'aarch64-linux-android',
    (.android, .arm) => 'armv7-linux-androideabi',
    (.android, .x64) => 'x86_64-linux-android',
    (.iOS, .arm64) =>
      codeConfig.iOS.targetSdk == .iPhoneSimulator
          ? 'aarch64-apple-ios-sim'
          : 'aarch64-apple-ios',
    (.iOS, .x64) => 'x86_64-apple-ios',
    _ => throw UnsupportedError('Unsupported target: $os $arch'),
  };
}

String _getLibraryFileName(OS os, String crateName) {
  if (os == .windows) {
    return '$crateName.dll';
  }

  if (os == .macOS || os == .iOS) {
    return 'lib$crateName.dylib';
  }

  return 'lib$crateName.so';
}

String _parseCrateName(String manifestPath) {
  final lines = File(manifestPath).readAsLinesSync();
  var inPackageSection = false;
  for (final line in lines) {
    final trimmed = line.trim();
    if (trimmed == '[package]') {
      inPackageSection = true;
      continue;
    }

    if (trimmed.startsWith('[')) {
      inPackageSection = false;
    }

    if (inPackageSection && trimmed.startsWith('name')) {
      final parts = trimmed.split('=');
      if (parts.length >= 2) {
        return parts[1].trim().replaceAll('"', '').replaceAll("'", '');
      }
    }
  }

  throw Exception('Could not find crate name in Cargo.toml at $manifestPath');
}

Map<String, String> _getBuildEnvVars(CodeConfig codeConfig) {
  final env = <String, String>{};
  final targetOS = codeConfig.targetOS;

  if (targetOS == .android) {
    final cCompiler = codeConfig.cCompiler;
    if (cCompiler != null) {
      final targetTriple = _mapToTriple(codeConfig);
      final targetTripleEnvVar = targetTriple.replaceAll('-', '_');

      // Deduction logic for NDK tools based on the compiler path
      final compilerPath = cCompiler.compiler.toFilePath();
      final binDir = p.dirname(compilerPath);

      String getBinaryPath(String name) {
        final binaryName = (Platform.isWindows) ? '$name.cmd' : name;
        final fullPath = p.join(binDir, binaryName);
        return File(fullPath).existsSync() ? fullPath : name;
      }

      final ndkTargetTriple = switch (targetTriple) {
        'armv7-linux-androideabi' => 'armv7a-linux-androideabi',
        _ => targetTriple,
      };

      // Dynamically fetch target API or default to 21 (Android 5.0)
      int apiTarget;
      try {
        apiTarget = codeConfig.android.targetNdkApi;
      } on Object catch (_) {
        apiTarget = 21;
      }

      final clangPath = getBinaryPath('$ndkTargetTriple$apiTarget-clang');
      final clangPpPath = getBinaryPath('$ndkTargetTriple$apiTarget-clang++');
      final ranlibPath = (Platform.isWindows)
          ? getBinaryPath('llvm-ranlib.exe')
          : getBinaryPath('llvm-ranlib');

      env['AR_$targetTripleEnvVar'] = cCompiler.archiver.toFilePath();
      env['CC_$targetTripleEnvVar'] = clangPath;
      env['CXX_$targetTripleEnvVar'] = clangPpPath;
      env['RANLIB_$targetTripleEnvVar'] = ranlibPath;
      env['CARGO_TARGET_${targetTripleEnvVar.toUpperCase()}_LINKER'] =
          clangPath;

      // Bindgen support (optional but good practice)
      final ndkToolchainRoot = p.dirname(p.dirname(clangPath));
      final sysroot = p.join(ndkToolchainRoot, 'sysroot');
      final ndkSysrootTargetTriple = switch (targetTriple) {
        'armv7-linux-androideabi' => 'arm-linux-androideabi',
        _ => targetTriple,
      };
      final extraInclude = '$sysroot/usr/include/$ndkSysrootTargetTriple';
      env['BINDGEN_EXTRA_CLANG_ARGS_$targetTripleEnvVar'] =
          '--sysroot=$sysroot -I$extraInclude'.replaceAll(r'\', '/');
    }
  }

  if (targetOS == OS.macOS) {
    try {
      env['MACOSX_DEPLOYMENT_TARGET'] = codeConfig.macOS.targetVersion
          .toString();
    } on Object catch (_) {}

    if (Platform.isMacOS) {
      final path = Platform.environment['PATH'];
      if (path != null) {
        env['PATH'] = path
            .split(':')
            .where((e) => !e.contains('Contents/Developer/'))
            .join(':');
      }
    }
  }

  if (targetOS == .iOS) {
    try {
      env['IPHONEOS_DEPLOYMENT_TARGET'] = codeConfig.iOS.targetVersion
          .toString();
    } on Object catch (_) {}
  }

  return env;
}

/// The source files a cargo dep-info file lists, as absolute URIs.
Iterable<Uri> _parseDependencies(String dependencyFilePath) {
  final file = File(dependencyFilePath);
  if (!file.existsSync()) {
    return const [];
  }

  return parseDependencyPaths(file.readAsStringSync()).map(Uri.file);
}

/// The dependency paths listed in the body of a cargo dep-info file.
///
/// Separated from the file reading so it can be tested against the platform it
/// is NOT running on: the Windows case below cannot be produced by a cargo run
/// on a Mac, and it is the one that fails silently.
///
/// The format is Makefile-like — one `target: dep dep dep` line per artifact —
/// which decides all three things this has to get right.
///
/// **Split on the first `": "`, not on the first `':'`.** A Windows path opens
/// with a drive letter, so `C:\out\m2v_ffi.dll: C:\src\lib.rs` cut at any colon
/// yields `\out\m2v_ffi.dll` as a "dependency" — a path that does not exist,
/// with the hook still reporting success. That bug was dormant while the caller
/// asked for a filename cargo never writes: the reader above returned early on
/// a missing file and never reached this parser at all. Fixing the caller is
/// what would have exposed it, on Windows only, as a hook that rebuilds when it
/// need not and skips rebuilds when it must not.
///
/// **Read every line.** A crate built as `["staticlib", "cdylib"]` gets a line
/// per artifact; taking only the first drops the rest.
///
/// **Respect escaped spaces.** Cargo writes a space inside a path as `\ `, so
/// splitting on bare whitespace tears `/Users/Jane Doe/src/lib.rs` in half.
List<String> parseDependencyPaths(String content) {
  final deps = <String>{};

  for (final line in const LineSplitter().convert(content)) {
    final separator = line.indexOf(': ');
    if (separator == -1) {
      continue;
    }

    final listed = line.substring(separator + 2).trim();
    for (final entry in listed.split(RegExp(r'(?<!\\)\s+'))) {
      // A lone `\` is a line continuation, not a path.
      if (entry.isEmpty || entry == r'\') {
        continue;
      }
      deps.add(entry.replaceAll(r'\ ', ' '));
    }
  }

  return deps.toList();
}
