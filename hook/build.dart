import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:path/path.dart' as p;

void main(List<String> args) async {
  await build(args, (input, output) async {
    final packageRoot = input.packageRoot.toFilePath();
    final nativeDir = p.join(packageRoot, 'native');

    final codeConfig = input.config.code;
    final targetOS = codeConfig.targetOS;
    final targetArch = codeConfig.targetArchitecture;

    // We default to release mode
    const buildMode = 'release';
    final targetTriple = _mapToTriple(targetOS, targetArch);

    final linkMode = codeConfig.linkModePreference == .static
        ? StaticLinking()
        : DynamicLoadingBundled();

    // 2. Execute Cargo Build
    // Optimization: Don't use --target if it's the host architecture
    // to avoid Cargo path complexity (target/release vs target/triple/release)
    final isHost = _isHost(targetOS, targetArch);

    stdout.writeln('Building Rust for $targetTriple (isHost: $isHost)...');

    final result = await Process.run(
      'cargo',
      [
        'build',
        if (buildMode == 'release') '--release',
        if (!isHost) ...['--target', targetTriple],
      ],
      workingDirectory: nativeDir,
    );

    if (result.exitCode != 0) {
      throw Exception('Rust build failed:\n${result.stderr}');
    }

    // 3. Identify Artifact
    const crateName = 'm2v_ffi';
    final libFileName = _getLibraryFileName(targetOS, crateName);

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
        file: Uri.file(binaryPath),
      ),
    );

    output.dependencies.add(Uri.directory(nativeDir));
  });
}

bool _isHost(OS os, Architecture arch) {
  if (Platform.isLinux && os == OS.linux && arch == Architecture.x64) {
    return true;
  }

  if (Platform.isMacOS && os == OS.macOS) {
    return true;
  }

  if (Platform.isWindows && os == OS.windows) {
    return true;
  }

  return false;
}

String _mapToTriple(OS os, Architecture arch) => switch ((os, arch)) {
  (OS.linux, Architecture.x64) => 'x86_64-unknown-linux-gnu',
  (OS.linux, Architecture.arm64) => 'aarch64-unknown-linux-gnu',
  (OS.macOS, Architecture.x64) => 'x86_64-apple-darwin',
  (OS.macOS, Architecture.arm64) => 'aarch64-apple-darwin',
  (OS.windows, Architecture.x64) => 'x86_64-pc-windows-msvc',
  (OS.windows, Architecture.arm64) => 'aarch64-pc-windows-msvc',
  (OS.android, Architecture.arm64) => 'aarch64-linux-android',
  (OS.android, Architecture.arm) => 'armv7-linux-androideabi',
  (OS.android, Architecture.x64) => 'x86_64-linux-android',
  (OS.iOS, Architecture.arm64) => 'aarch64-apple-ios',
  _ => throw UnsupportedError('Unsupported target: $os $arch'),
};

String _getLibraryFileName(OS os, String crateName) {
  if (os == OS.windows) {
    return '$crateName.dll';
  }

  if (os == OS.macOS || os == OS.iOS) {
    return 'lib$crateName.dylib';
  }

  return 'lib$crateName.so';
}
