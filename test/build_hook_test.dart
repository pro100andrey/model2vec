import 'dart:convert';
import 'dart:ffi' show Abi;
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:test/test.dart';

import '../hook/build.dart';

/// The build hook is invoked by more than one embedder, and they do not all ask
/// for the same assets.
///
/// `dart build` only ever runs it with `code_assets/code` in
/// `build_asset_types`. `flutter run` also runs it from the asset-bundling pass
/// — `FlutterHookRunnerNative.runHooks`, which passes `buildCodeAssets: null` —
/// and with data assets behind a feature flag that pass arrives with the list
/// **empty**. Reading `input.config.code` there throws `StateError`, the hook
/// exits 255, and Flutter reports "Building native assets failed" for the whole
/// app: the pass that would have built the library never gets to run.
///
/// This test pins the empty case, which is the one no local `dart` command
/// reaches. It builds nothing, so it needs neither cargo nor a model and stays
/// in the fast lane.
void main() {
  _dependencyParsing();

  group('build hook', () {
    late Directory work;

    setUp(() => work = Directory.systemTemp.createTempSync('m2v_hook_'));
    tearDown(() => work.deleteSync(recursive: true));

    /// Run `hook/build.dart` against a config asking for [buildAssetTypes],
    /// the way the hook runner does: a JSON file passed as `--config`.
    Future<ProcessResult> runHook(
      List<String> buildAssetTypes, {
      Map<String, Object?>? codeConfig,
    }) {
      final outFile = p.join(work.path, 'output.json');
      final sharedDir = p.join(work.path, 'shared');
      final configFile = File(p.join(work.path, 'input.json'))
        ..writeAsStringSync(
          jsonEncode({
            'assets': <String, Object?>{},
            'config': {
              'build_asset_types': buildAssetTypes,
              'linking_enabled': false,
              if (codeConfig != null) 'extensions': {'code_assets': codeConfig},
            },
            'out_dir_shared': sharedDir,
            'out_file': outFile,
            'package_name': 'model2vec',
            'package_root': '${Directory.current.path}${p.separator}',
            'user_defines': <String, Object?>{},
          }),
        );

      return Process.run(Platform.resolvedExecutable, [
        p.join('hook', 'build.dart'),
        '--config=${configFile.path}',
      ]);
    }

    test('an invocation asking for no assets at all succeeds', () async {
      final result = await runHook([]);

      expect(
        result.exitCode,
        0,
        reason:
            'the hook must not read `input.config.code` unless the invoker '
            'asked for code assets:\n${result.stderr}',
      );
      expect(result.stderr, isNot(contains('HookConfig.code')));
    });

    test('an invocation asking for no assets writes an empty output', () async {
      await runHook([]);

      final output = File(p.join(work.path, 'output.json'));
      expect(
        output.existsSync(),
        isTrue,
        reason: 'the hook runner reads this file back and fails without it',
      );

      final decoded = jsonDecode(output.readAsStringSync()) as Map;
      final assets = (decoded['assets'] as Map?) ?? const {};
      expect(
        assets['code_assets'] ?? const <Object?>[],
        isEmpty,
        reason: 'nothing was asked for, so nothing may be registered',
      );
    });

    /// What the hook declares it depends on — the crate's sources, never the
    /// directory it builds into.
    ///
    /// A hook whose declared input contains its own output can never be
    /// cached: the runner sees the input change on every build, reports
    /// `File modified during build. Build must be rerun.`, and hashes the tree
    /// again. `native/` holds cargo's `target/`, which reaches hundreds of
    /// megabytes, so the cost is not theoretical — see dart-lang/native#1998
    /// for the same shape reported against `native_toolchain_c`.
    ///
    /// This asks for code assets, so unlike its neighbours above it needs
    /// cargo. Hence the tag: `dart test -x integration` keeps the fast lane
    /// fast, `dart test -t integration` runs this.
    test(
      'the declared dependencies are the sources, not the build tree',
      () async {
        final result = await runHook(
          const ['code_assets/code'],
          codeConfig: _hostCodeConfig(),
        );
        expect(result.exitCode, 0, reason: 'stderr: ${result.stderr}');

        final decoded =
            jsonDecode(
                  File(p.join(work.path, 'output.json')).readAsStringSync(),
                )
                as Map;
        final declared = ((decoded['dependencies'] as List?) ?? const [])
            .cast<String>();

        // Compared with separators normalised, so the one assertion covers
        // the Windows paths this hook also produces.
        final paths = [for (final d in declared) d.replaceAll(r'\', '/')];

        // Cargo names its dep-info after the TARGET (`libm2v_ffi.d`), not
        // after the artifact (`libm2v_ffi.dylib`). Asking for the artifact
        // path plus `.d` names a file cargo never writes, so the parse yields
        // nothing and the fallback is taken every single time. These two
        // sources are the proof the dep-info was found and actually read.
        for (final source in const ['lib.rs', 'model.rs']) {
          expect(
            paths.any((d) => d.endsWith('src/$source')),
            isTrue,
            reason:
                'the cargo dep-info was not parsed — $source is missing '
                'from $paths',
          );
        }

        // The regression itself: `native/` as a bare directory. It is the
        // parent of `target/`, so declaring it makes every build dirty the
        // next one.
        expect(
          paths.where((d) => d.endsWith('native/')),
          isEmpty,
          reason:
              'the build tree is declared as an input, so the hook '
              'invalidates itself on every run: $paths',
        );
      },
      tags: ['integration'],
      timeout: const Timeout(Duration(minutes: 10)),
    );
  });
}

/// The dep-info parser, against inputs this machine cannot itself produce.
///
/// The integration test above covers the host's own dep-info. It cannot cover
/// the Windows one, and the Windows one is the dangerous case: a drive letter
/// is a colon, so cutting the line at the first colon returns paths that do not
/// exist while the hook still reports success. Pure string in, list out, so the
/// platform running the test does not matter.
void _dependencyParsing() {
  group('cargo dep-info parsing', () {
    test('a Windows line is cut after the target, not at the drive', () {
      final deps = parseDependencyPaths(
        r'C:\proj\native\target\release\m2v_ffi.dll: '
        r'C:\proj\native\src\lib.rs C:\proj\native\src\model.rs',
      );

      expect(deps, [
        r'C:\proj\native\src\lib.rs',
        r'C:\proj\native\src\model.rs',
      ]);
      // The shape of the old bug: the drive letter taken for the separator,
      // leaving a rootless path that matches nothing on disk.
      expect(deps, isNot(contains(r'\proj\native\target\release\m2v_ffi.dll')));
    });

    test('every artifact line contributes, not just the first', () {
      final deps = parseDependencyPaths(
        '/p/target/release/libm2v_ffi.a: /p/src/lib.rs\n'
        '/p/target/release/libm2v_ffi.dylib: /p/src/model.rs\n',
      );

      expect(deps, containsAll(['/p/src/lib.rs', '/p/src/model.rs']));
    });

    test('a space inside a path is one path, not two', () {
      final deps = parseDependencyPaths(
        r'/p/out.dylib: /Users/Jane\ Doe/src/lib.rs /p/src/model.rs',
      );

      expect(deps, contains('/Users/Jane Doe/src/lib.rs'));
      expect(deps, isNot(contains('/Users/Jane')));
    });

    test('a file with no dependency line yields nothing', () {
      expect(parseDependencyPaths('not a depfile\n'), isEmpty);
    });
  });
}

/// The `code_assets` config block for the machine running the test.
///
/// The hook only skips `rustup target add` and reads `target/release/` — rather
/// than `target/<triple>/release/` — when the requested target IS the host, so
/// a hard-coded triple would test the cross path on every other machine.
Map<String, Object?> _hostCodeConfig() {
  final arch = switch (Abi.current()) {
    Abi.macosArm64 || Abi.linuxArm64 || Abi.windowsArm64 => 'arm64',
    Abi.macosX64 || Abi.linuxX64 || Abi.windowsX64 => 'x64',
    _ => throw UnsupportedError('unmapped host ABI ${Abi.current()}'),
  };

  return {
    'link_mode_preference': 'dynamic',
    'target_architecture': arch,
    'target_os': Platform.operatingSystem,
    if (Platform.isMacOS) 'macos': {'target_version': 12},
  };
}
