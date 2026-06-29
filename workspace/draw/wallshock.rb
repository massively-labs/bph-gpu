require "gnuplot"

class Wallshock
  def self.draw
    f = open("dat/wallshock.dat")
    content = f.read.split("\n")

    size = content.size
    step = 1.0 / size
    x = (0...size).map { |i| i * step + 0.5 * step }
    y = content

    Gnuplot.open do |gp|
      Gnuplot::Plot.new(gp) do |plot|
        plot.terminal "jpeg"
        plot.output "plot/wallshock.jpeg"
        plot.xrange "[0:0.3]"
        plot.yrange "[0.5:6.5]"
        plot.xlabel "x"
        plot.ylabel "density"

        plot.data << Gnuplot::DataSet.new([x, y]) do |ds|
          ds.with = "lines"
          ds.title = "wallshock"
        end
      end
    end
  end
end
