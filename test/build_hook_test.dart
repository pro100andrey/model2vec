import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:test/test.dart';

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
  group('build hook', () {
    late Directory work;

    setUp(() => work = Directory.systemTemp.createTempSync('m2v_hook_'));
    tearDown(() => work.deleteSync(recursive: true));

    /// Run `hook/build.dart` against a config asking for [buildAssetTypes],
    /// the way the hook runner does: a JSON file passed as `--config`.
    Future<ProcessResult> runHook(List<String> buildAssetTypes) {
      final outFile = p.join(work.path, 'output.json');
      final sharedDir = p.join(work.path, 'shared');
      final configFile = File(p.join(work.path, 'input.json'))
        ..writeAsStringSync(
          jsonEncode({
            'assets': <String, Object?>{},
            'config': {
              'build_asset_types': buildAssetTypes,
              'linking_enabled': false,
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
  });
}
