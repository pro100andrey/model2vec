import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;

void main(List<String> args) async {
  await build(args, (input, output) async {
    final packageRoot = input.packageRoot.toFilePath();
    final nativeDir = p.join(packageRoot, 'native');
    final manifestPath = p.join(nativeDir, 'Cargo.toml');
    final crateName = _parseCrateName(manifestPath);

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
        ? p.join(nativeDir, 'target', buildMode, libFileName)
        : p.join(nativeDir, 'target', targetTriple, buildMode, libFileName);

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
    final depFilePath = '$binaryPath.d';
    final deps = _parseDependencies(depFilePath);
    if (deps.isNotEmpty) {
      output.dependencies.addAll(deps);
    } else {
      // Fallback if .d file is missing
      output.dependencies.add(.directory(nativeDir));
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

Iterable<Uri> _parseDependencies(String dependencyFilePath) {
  if (!File(dependencyFilePath).existsSync()) {
    return [];
  }

  final content = File(dependencyFilePath).readAsStringSync();
  final parts = content.split(':');
  if (parts.length < 2) {
    return [];
  }

  // The first part is the target file, second part are dependencies
  return parts[1]
      .trim()
      .split(RegExp(r'\s+'))
      .where((e) => e.isNotEmpty && e != r'\')
      .map(Uri.file);
}
