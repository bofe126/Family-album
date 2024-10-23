import 'package:flutter/material.dart';
import 'face_recognition_service.dart';
import 'dart:typed_data';
import 'package:provider/provider.dart';
import 'photo_state.dart';
import 'logger.dart';  // 添加 logger 导入

class FaceRecognition extends StatefulWidget {
  final List<String> imagePaths;
  const FaceRecognition({Key? key, required this.imagePaths}) : super(key: key);

  @override
  _FaceRecognitionState createState() => _FaceRecognitionState();
}

class _FaceRecognitionState extends State<FaceRecognition> {
  bool _isRecognizing = false;
  List<Uint8List> _detectedFaces = [];
  bool _showAllDetections = false;
  final _logger = logger;  // 使用导入的 logger

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _recognizeFaces();
    });
  }

  Future<void> _recognizeFaces() async {
    try {
      if (widget.imagePaths.isEmpty) {
        final photoState = Provider.of<PhotoState>(context, listen: false);
        widget.imagePaths
            .addAll(photoState.imageFileList.map((file) => file.path));
      }

      if (widget.imagePaths.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('没有可识别的照片')),
        );
        return;
      }

      setState(() {
        _isRecognizing = true;
        _detectedFaces = [];
      });

      _logger.i("Starting face recognition for ${widget.imagePaths.length} images");
      for (var path in widget.imagePaths) {
        _logger.i("Processing image: $path");
      }

      List<FaceData> faceDataList =
          await FaceRecognitionService.recognizeFaces(widget.imagePaths);

      setState(() {
        _isRecognizing = false;
        _detectedFaces = faceDataList.map((faceData) => faceData.faceImage).toList();
      });

      if (_detectedFaces.length >= 100) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('检测到100张或更多人脸。可能有些人脸未被处理。')),
        );
      }

      _logger.i("人脸识别完成，共检测到 ${_detectedFaces.length} 个人脸");
    } catch (e) {
      _logger.e("_recognizeFaces 出错: $e");
      setState(() {
        _isRecognizing = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('人脸识别'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: _isRecognizing ? null : _recognizeFaces,
                    child: Text(_isRecognizing ? '正在识别...' : '重新识别'),
                  ),
                  const SizedBox(height: 20),
                  Text('图片数量: ${widget.imagePaths.length}'),
                  if (_detectedFaces.isNotEmpty)
                    Text('检测到的人脸数量: ${_detectedFaces.length}'),
                  if (_isRecognizing) const CircularProgressIndicator(),
                  CheckboxListTile(
                    title: const Text("显示所有检测结果"),
                    value: _showAllDetections,
                    onChanged: (newValue) {
                      setState(() {
                        _showAllDetections = newValue!;
                      });
                    },
                  ),
                ],
              ),
            ),
          ),
          SizedBox(
            height: 100,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: _showAllDetections
                  ? _detectedFaces.length
                  : _detectedFaces.length ~/ 2,
              itemBuilder: (context, index) {
                return Padding(
                  padding: const EdgeInsets.all(4.0),
                  child: Image.memory(_detectedFaces[index]),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
