import 'package:flutter/material.dart';
import 'face_recognition_service.dart';
import 'dart:typed_data';
import 'package:provider/provider.dart';
import 'photo_state.dart';

class FaceRecognition extends StatefulWidget {
  final List<String> imagePaths;
  const FaceRecognition({Key? key, required this.imagePaths}) : super(key: key);

  @override
  // ignore: library_private_types_in_public_api
  _FaceRecognitionState createState() => _FaceRecognitionState();
}

class _FaceRecognitionState extends State<FaceRecognition> {
  bool _isRecognizing = false;
  List<Uint8List> _detectedFaces = [];
  bool _showAllDetections = false;

  @override
  void initState() {
    super.initState();
    // 使用 WidgetsBinding.instance.addPostFrameCallback 来确保在 widget 树构建完成后执行
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _recognizeFaces();
    });
  }

  Future<void> _recognizeFaces() async {
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

    print("Starting face recognition for ${widget.imagePaths.length} images");
    for (var path in widget.imagePaths) {
      print("Processing image: $path");
    }

    List<FaceData> faceDataList =
        await FaceRecognitionService.recognizeFaces(widget.imagePaths);

    setState(() {
      _isRecognizing = false;
      _detectedFaces = faceDataList.map((faceData) => faceData.faceImage).toList();
    });

    if (_detectedFaces.length >= 100) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('检测到100张或更多人脸。可能有些人脸未被处理。')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('人脸识别'),
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
                  SizedBox(height: 20),
                  Text('图片数量: ${widget.imagePaths.length}'),
                  if (_detectedFaces.isNotEmpty)
                    Text('检测到的人脸数量: ${_detectedFaces.length}'),
                  if (_isRecognizing) CircularProgressIndicator(),
                  CheckboxListTile(
                    title: Text("显示所有检测结果"),
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
          Container(
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
