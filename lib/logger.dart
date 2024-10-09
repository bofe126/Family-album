import 'package:logger/logger.dart';
import 'package:path/path.dart' as path;

class CustomPrinter extends LogPrinter {
  final bool colors;
  final bool printTime;

  CustomPrinter({this.colors = true, this.printTime = true});

  @override
  List<String> log(LogEvent event) {
    var color = _getLevelColor(event.level);
    var levelLabel = _getLevelLabel(event.level);
    var time = printTime ? _getTime() : '';
    var fileInfo = _getFileInfo(StackTrace.current);

    var output = '$time $levelLabel $fileInfo ${event.message}';

    if (colors) {
      output = color(output);
    }

    return [output];
  }

  String _getTime() {
    var now = DateTime.now();
    return '${now.year}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')} '
        '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}:${now.second.toString().padLeft(2, '0')}';
  }

  String _getFileInfo(StackTrace stackTrace) {
    var frames = stackTrace.toString().split('\n');
    for (var frame in frames) {
      // 跳过不包含文件信息的帧
      if (!frame.contains('.dart')) continue;
      
      // 跳过 logger.dart 和 async 相关的帧
      if (frame.contains('logger.dart') || frame.contains('async')) continue;

      // 尝试匹配文件名和行号
      var match = RegExp(r'(\S+\.dart):(\d+):\d+').firstMatch(frame);
      if (match != null) {
        var fileName = path.basename(match.group(1)!);
        var lineNumber = match.group(2);
        return '[$fileName:$lineNumber]';
      }
    }
    return '[unknown]';  // 确保总是返回一个字符串
  }

  String _getLevelLabel(Level level) {
    switch (level) {
      case Level.verbose:
        return '[V]';
      case Level.debug:
        return '[D]';
      case Level.info:
        return '[I]';
      case Level.warning:
        return '[W]';
      case Level.error:
        return '[E]';
      case Level.wtf:
        return '[F]'; // F for Fatal
      default:
        return '[?]';
    }
  }

  AnsiColor _getLevelColor(Level level) {
    switch (level) {
      case Level.verbose:
        return AnsiColor.fg(AnsiColor.grey(0.5));
      case Level.debug:
        return AnsiColor.fg(6);
      case Level.info:
        return AnsiColor.fg(2);
      case Level.warning:
        return AnsiColor.fg(3);
      case Level.error:
        return AnsiColor.fg(196);
      case Level.wtf:
        return AnsiColor.fg(199);
      default:
        return AnsiColor.fg(7);
    }
  }
}

// 创建一个全局的 logger 实例
final logger = Logger(
  printer: CustomPrinter(colors: true, printTime: true),
);
