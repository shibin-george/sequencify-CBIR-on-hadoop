import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URI;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.imageio.ImageIO;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.util.LineReader;

public class BinaryFilesToHadoopSequenceFile {

	private static Log logger = LogFactory
			.getLog(BinaryFilesToHadoopSequenceFile.class);

	public static class ImageToSequenceFileMapper extends
			Mapper<Object, Text, Text, BytesWritable> {

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {

			logger.info("map method called..");

			String uri = value.toString();
			Configuration conf = new Configuration();
			FileSystem fs = FileSystem.get(URI.create(uri), conf);
			FSDataInputStream in = null;
			BufferedImage b;
			try {
				in = fs.open(new Path(uri));
				b = ImageIO.read(in);
				int height = b.getHeight();
				int width = b.getWidth();
				String s = "@@" + height + "X" + width + "@@";
				in.seek(0);
				BytesWritable v = new BytesWritable(
						org.apache.commons.io.IOUtils.toByteArray(in));
				// while( in.read(buffer, 0, buffer.length) >= 0 ) {
				// bout.write(buffer);
				// }
				s = value.toString() + s;
				value.set(s);
				context.write(value, v);
			} finally {
				IOUtils.closeStream(in);
			}
		}

	}

	public static class SequenceFilePartitioner extends
			Partitioner<Text, BytesWritable> {

		@Override
		public int getPartition(Text key, BytesWritable value,
				int numReduceTasks) {

			char r = key.toString().charAt(0);

			int target = r - 48;
			return target % numReduceTasks;

		}
	}

	public static class ImageToSequenceFileReducer extends
			Reducer<Text, IntWritable, Text, IntWritable> {

		// MultipleOutputs msfw;

		@Override
		public void reduce(Text key, Iterable<IntWritable> values,
				Context context) throws IOException, InterruptedException {
			// msfw.w
		}
	}

	public class NLinesRecordReader extends RecordReader<LongWritable, Text> {
		private int NLINESTOPROCESS = 1;
		private LineReader in;
		private LongWritable key;
		private Text value = new Text();
		private long start = 0;
		private long end = 0;
		private long pos = 0;
		private int maxLineLength;

		@Override
		public void close() throws IOException {
			if (in != null) {
				in.close();
			}
		}

		@Override
		public LongWritable getCurrentKey() throws IOException,
				InterruptedException {
			return key;
		}

		@Override
		public Text getCurrentValue() throws IOException, InterruptedException {
			return value;
		}

		@Override
		public float getProgress() throws IOException, InterruptedException {
			if (start == end) {
				return 0.0f;
			} else {
				return Math.min(1.0f, (pos - start) / (float) (end - start));
			}
		}

		@Override
		public void initialize(InputSplit genericSplit,
				TaskAttemptContext context) throws IOException,
				InterruptedException {
			FileSplit split = (FileSplit) genericSplit;
			final Path file = split.getPath();
			Configuration conf = context.getConfiguration();

			// get the parameter from the configuration
			NLINESTOPROCESS = Integer.parseInt(conf.get("NLINESTOPROCESS"));
			this.maxLineLength = conf.getInt(
					"mapred.linerecordreader.maxlength", Integer.MAX_VALUE);
			FileSystem fs = file.getFileSystem(conf);
			start = split.getStart();
			end = start + split.getLength();
			boolean skipFirstLine = false;
			FSDataInputStream filein = fs.open(split.getPath());

			if (start != 0) {
				skipFirstLine = true;
				--start;
				filein.seek(start);
			}
			in = new LineReader(filein, conf);
			if (skipFirstLine) {
				start += in.readLine(new Text(), 0,
						(int) Math.min((long) Integer.MAX_VALUE, end - start));
			}
			this.pos = start;
		}

		@Override
		public boolean nextKeyValue() throws IOException, InterruptedException {
			if (key == null) {
				key = new LongWritable();
			}
			key.set(pos);
			if (value == null) {
				value = new Text();
			}
			value.clear();
			final Text endline = new Text("\n");
			int newSize = 0;
			for (int i = 0; i < NLINESTOPROCESS; i++) {
				Text v = new Text();
				while (pos < end) {
					newSize = in.readLine(
							v,
							maxLineLength,
							Math.max((int) Math.min(Integer.MAX_VALUE, end
									- pos), maxLineLength));
					value.append(v.getBytes(), 0, v.getLength());
					value.append(endline.getBytes(), 0, endline.getLength());
					if (newSize == 0) {
						break;
					}
					pos += newSize;
					if (newSize < maxLineLength) {
						break;
					}
				}
			}
			if (newSize == 0) {
				key = null;
				value = null;
				return false;
			} else {
				return true;
			}
		}
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		// JobConf j;
		String[] otherArgs = new GenericOptionsParser(conf, args)
				.getRemainingArgs();
		if (otherArgs.length < 3) {
			System.err
					.println("Usage: BinaryFilesToHadoopSequenceFile <in Path for url file> "
							+ "<approximate number of image files> "
							+ "<number of sequence files to output>");
			System.exit(2);
		}

		Job job = new Job(conf, "BinaryFilesToHadoopSequenceFile");
		job.setJarByClass(BinaryFilesToHadoopSequenceFile.class);
		job.setMapperClass(ImageToSequenceFileMapper.class);
		// job.setOutputKeyClass(Text.class);
		// job.setOutputValueClass(BytesWritable.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		//
		// .addNamedOutput(job, "seq1", SequenceFileOutputFormat.class,
		// Text.class, BytesWritable.class);

		for (int i = 0; i < args.length - 1; i++) {
			// FileInputFormat.setInputPaths(job, new Path(args[i]));
			MultipleInputs.addInputPath(job, new Path(args[i]),
					TextInputFormat.class);
		}
		job.setPartitionerClass(SequenceFilePartitioner.class);
		job.setNumReduceTasks(10);
		DateFormat dateFormat = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");
		Date date = new Date();
		String outputfolder = "output-" + dateFormat.format(date);
		// FileOutputFormat f = new FileOutputFormat();
		// FileOutputFormat.setOutputName(job, filename);
		// job.getConfiguration().set(FileOutputFormat.BASE_OUTPUT_NAME,
		// filename);
		conf.set("BASE_OUTPUT_FILE_NAME", outputfolder);
		int numOutputFiles = Integer.parseInt(args.length)
		MultipleOutputs
				.addNamedOutput(job, filename, SequenceFileOutputFormat.class,
						Text.class, BytesWritable.class);

		FileOutputFormat.setOutputPath(job, new Path(outputfolder));
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}